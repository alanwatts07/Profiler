"""Audio capture module for recording system audio.

Captures audio from PulseAudio/PipeWire monitor devices.
"""

import logging
import threading
import queue
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pulsectl
    PULSECTL_AVAILABLE = True
except ImportError:
    PULSECTL_AVAILABLE = False

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float


class AudioCapture:
    """Captures audio from system audio output."""

    def __init__(
        self,
        device_name: Optional[str] = None,
        sample_rate: int = None,
        chunk_duration: float = None,
        callback: Optional[Callable[[AudioChunk], None]] = None
    ):
        """Initialize audio capture.

        Args:
            device_name: PulseAudio monitor device name (auto-detect if None)
            sample_rate: Sample rate in Hz (default from config)
            chunk_duration: Duration of each chunk in seconds
            callback: Function to call when a chunk is ready
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is required for audio capture")

        self.device_name = device_name or config.PULSEAUDIO_DEVICE
        self.sample_rate = sample_rate or config.AUDIO_SAMPLE_RATE
        self.chunk_duration = chunk_duration or config.AUDIO_CHUNK_DURATION
        self.callback = callback

        self._stream = None
        self._recording = False
        self._audio_queue = queue.Queue()
        self._buffer = []
        self._buffer_samples = 0
        self._chunk_samples = int(self.sample_rate * self.chunk_duration)
        self._thread = None
        self._start_time = None

    def detect_monitor_device(self) -> Optional[str]:
        """Find PulseAudio/PipeWire monitor source.

        Returns:
            Device name or None if not found
        """
        if not PULSECTL_AVAILABLE:
            logger.warning("pulsectl not available, using default device")
            return None

        try:
            with pulsectl.Pulse('profiler-detect') as pulse:
                # Look for monitor sources
                for source in pulse.source_list():
                    if '.monitor' in source.name:
                        logger.info(f"Found monitor device: {source.name}")
                        return source.name

                # Fallback: look for any source with monitor in description
                for source in pulse.source_list():
                    if source.description and 'monitor' in source.description.lower():
                        logger.info(f"Found monitor device: {source.name}")
                        return source.name

            logger.warning("No monitor device found")
            return None
        except Exception as e:
            logger.error(f"Error detecting monitor device: {e}")
            return None

    def list_devices(self, loopback=False) -> list:
        """List available audio devices.

        Args:
            loopback: If True, list output devices for loopback capture (Windows)

        Returns:
            List of device dictionaries
        """
        devices = []

        if SOUNDDEVICE_AVAILABLE:
            try:
                device_list = sd.query_devices()
                for i, dev in enumerate(device_list):
                    # For loopback, we want OUTPUT devices (to capture system audio)
                    if loopback:
                        if dev['max_output_channels'] > 0:
                            devices.append({
                                'index': i,
                                'name': dev['name'],
                                'channels': dev['max_output_channels'],
                                'sample_rate': dev['default_samplerate'],
                                'type': 'output',
                                'hostapi': dev.get('hostapi', 0),
                            })
                    else:
                        # Regular input devices
                        if dev['max_input_channels'] > 0:
                            devices.append({
                                'index': i,
                                'name': dev['name'],
                                'channels': dev['max_input_channels'],
                                'sample_rate': dev['default_samplerate'],
                                'type': 'input',
                                'hostapi': dev.get('hostapi', 0),
                            })
            except Exception as e:
                logger.error(f"Error listing devices: {e}")

        return devices

    def list_loopback_devices(self) -> list:
        """List output devices that can be used for loopback capture.

        On Windows, use WASAPI loopback to capture system audio.

        Returns:
            List of output device dictionaries
        """
        return self.list_devices(loopback=True)

    def start_recording(self):
        """Start continuous audio capture."""
        if self._recording:
            logger.warning("Already recording")
            return

        # Determine device to use
        device = None
        if self.device_name is None or self.device_name == 'auto':
            # Auto-detect monitor device
            devices = self.list_devices()
            for dev in devices:
                if 'monitor' in dev['name'].lower():
                    device = dev['index']
                    logger.info(f"Auto-selected monitor device: {device} ({dev['name']})")
                    break
            if device is None:
                logger.warning("No monitor device found, using default")
        elif isinstance(self.device_name, int):
            # Device specified by index
            device = self.device_name
        else:
            # Device specified by name - find index
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if self.device_name in dev['name']:
                    device = i
                    break

        logger.info(f"Starting audio capture on device: {device}")

        # Query device info to get channel count and sample rate
        device_info = sd.query_devices(device)
        device_channels = int(device_info['max_input_channels'])
        if device_channels == 0:
            # This is an output-only device, try max_output_channels
            device_channels = int(device_info['max_output_channels'])
        if device_channels == 0:
            device_channels = 2  # Default to stereo

        # Use device's default sample rate if ours isn't supported
        device_sample_rate = int(device_info['default_samplerate'])

        # Try the requested sample rate first, fall back to device default
        logger.info(f"Device has {device_channels} channels, default sample rate: {device_sample_rate}")
        self._device_channels = device_channels

        self._recording = True
        self._buffer = []
        self._buffer_samples = 0
        self._start_time = time.time()

        # Try requested sample rate, fall back to device default
        for try_sample_rate in [self.sample_rate, device_sample_rate, 48000, 44100]:
            try:
                self._stream = sd.InputStream(
                    device=device,
                    samplerate=try_sample_rate,
                    channels=device_channels,
                    dtype='float32',
                    callback=self._audio_callback,
                    blocksize=int(try_sample_rate * 0.1),  # 100ms blocks
                )
                self._stream.start()
                self.sample_rate = try_sample_rate  # Update to actual rate used
                logger.info(f"Using sample rate: {try_sample_rate}")
                break
            except Exception as e:
                logger.warning(f"Sample rate {try_sample_rate} failed: {e}")
                continue
        else:
            self._recording = False
            raise RuntimeError("Could not open audio stream with any sample rate")

        # Start processing thread
        self._thread = threading.Thread(target=self._process_audio, daemon=True)
        self._thread.start()

        logger.info("Audio capture started")

    def stop_recording(self):
        """Stop audio capture gracefully."""
        if not self._recording:
            return

        logger.info("Stopping audio capture")
        self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Process any remaining buffer
        if self._buffer:
            self._flush_buffer()

        logger.info("Audio capture stopped")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio status: {status}")

        if self._recording:
            self._audio_queue.put(indata.copy())

    def _process_audio(self):
        """Process audio from queue in background thread."""
        while self._recording or not self._audio_queue.empty():
            try:
                data = self._audio_queue.get(timeout=0.5)
                self._add_to_buffer(data)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

    def _add_to_buffer(self, data: np.ndarray):
        """Add audio data to buffer and emit chunks when ready."""
        # Convert multi-channel to mono by averaging channels
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        self._buffer.append(data)
        self._buffer_samples += len(data)

        while self._buffer_samples >= self._chunk_samples:
            self._emit_chunk()

    def _emit_chunk(self):
        """Emit a complete chunk."""
        if not self._buffer:
            return

        # Concatenate buffer
        all_data = np.concatenate(self._buffer)

        # Extract chunk
        chunk_data = all_data[:self._chunk_samples].flatten()

        # Keep remainder in buffer
        remainder = all_data[self._chunk_samples:]
        self._buffer = [remainder] if len(remainder) > 0 else []
        self._buffer_samples = len(remainder)

        # Create chunk object
        chunk = AudioChunk(
            data=chunk_data,
            sample_rate=self.sample_rate,
            timestamp=time.time() - self._start_time,
            duration=self.chunk_duration
        )

        # Call callback
        if self.callback:
            try:
                self.callback(chunk)
            except Exception as e:
                logger.error(f"Error in chunk callback: {e}")

    def _flush_buffer(self):
        """Flush remaining buffer as partial chunk."""
        if not self._buffer:
            return

        all_data = np.concatenate(self._buffer)
        if len(all_data) > self.sample_rate * 0.5:  # At least 0.5 seconds
            chunk = AudioChunk(
                data=all_data.flatten(),
                sample_rate=self.sample_rate,
                timestamp=time.time() - self._start_time,
                duration=len(all_data) / self.sample_rate
            )
            if self.callback:
                try:
                    self.callback(chunk)
                except Exception as e:
                    logger.error(f"Error in flush callback: {e}")

        self._buffer = []
        self._buffer_samples = 0

    def save_chunk(self, chunk: AudioChunk, filepath: str):
        """Save audio chunk as WAV file.

        Args:
            chunk: AudioChunk to save
            filepath: Output file path
        """
        if not SCIPY_AVAILABLE:
            logger.error("scipy is required to save audio files")
            return

        # Convert to int16 for WAV
        audio_int16 = (chunk.data * 32767).astype(np.int16)
        wavfile.write(filepath, chunk.sample_rate, audio_int16)
        logger.debug(f"Saved audio chunk to {filepath}")

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
