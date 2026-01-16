"""Command-line interface for the Audio Profiling System.

Usage:
    profiler record start        - Start recording system audio
    profiler record stop         - Stop recording
    profiler speakers list       - List all speakers
    profiler speakers profile ID - View speaker profile
    profiler speakers rename OLD NEW - Rename speaker
    profiler sessions list       - List sessions
    profiler sessions view ID    - View session transcript
    profiler process FILE        - Process audio file
    profiler analyze TEXT        - Analyze text directly
"""

import sys
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from .config import config, DATA_DIR
from .storage.database import init_db, get_db
from .storage.models import Speaker, SpeakerProfile, Session, Utterance
from .profiling.behavioral_profiler import BehavioralProfiler
from .profiling.linguistic_analyzer import LinguisticAnalyzer

console = Console()
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler(sys.stderr)
        ]
    )


@click.group()
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def cli(debug):
    """Audio Profiling System - Behavioral analysis from speech."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


# =============================================================================
# Helper Functions
# =============================================================================

def _display_analysis(analysis, text=None):
    """Display analysis results in a nice format."""
    # VAK
    console.print("\n[bold cyan]VAK Modality:[/bold cyan]")
    dominant_vak = analysis.get_dominant_vak()
    for modality, score in analysis.vak_scores.items():
        marker = "→" if modality == dominant_vak else " "
        bar = "█" * int(score * 20)
        console.print(f"  {marker} {modality:12}: {bar} {score:.2f}")

    # Social Needs
    console.print("\n[bold cyan]Social Needs:[/bold cyan]")
    sorted_needs = sorted(analysis.social_needs.items(), key=lambda x: x[1], reverse=True)
    for need, score in sorted_needs[:4]:
        bar = "█" * int(score * 20)
        style = "bold green" if score > 0.5 else "white"
        console.print(f"  [{style}]{need:12}: {bar} {score:.2f}[/{style}]")

    # Decision Styles
    if analysis.decision_styles:
        console.print(f"\n[bold cyan]Decision Styles:[/bold cyan] {', '.join(analysis.decision_styles)}")

    # Certainty
    cert = analysis.communication_patterns.get("certainty", 0.5)
    cert_label = "High" if cert > 0.6 else "Low" if cert < 0.4 else "Moderate"
    console.print(f"\n[bold cyan]Certainty:[/bold cyan] {cert_label} ({cert:.2f})")

    # Influence patterns
    if analysis.influence_patterns:
        console.print(f"[bold cyan]Influence Tactics:[/bold cyan] {', '.join(analysis.influence_patterns)}")

    # Emotional indicators
    if analysis.emotional_indicators:
        top_emotions = sorted(analysis.emotional_indicators.items(), key=lambda x: x[1], reverse=True)[:2]
        top_emotions = [(e, s) for e, s in top_emotions if s > 0]
        if top_emotions:
            emotions_str = ", ".join(f"{e} ({s:.2f})" for e, s in top_emotions)
            console.print(f"[bold cyan]Emotions:[/bold cyan] {emotions_str}")


# =============================================================================
# Recording Commands
# =============================================================================

@cli.group()
def record():
    """Recording controls."""
    pass


@record.command('devices')
@click.option('--all', '-a', 'show_all', is_flag=True, help='Show all devices (input and output)')
def record_devices(show_all):
    """List available audio devices for recording."""
    from .audio.audio_capture import AudioCapture
    import sounddevice as sd

    capture = AudioCapture()

    # Get all devices raw
    all_devices = sd.query_devices()

    # Input devices (mics, Stereo Mix, etc.)
    console.print("\n[bold cyan]INPUT DEVICES (microphones, Stereo Mix):[/bold cyan]")
    table = Table(box=box.ROUNDED)
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="green")
    table.add_column("Channels", justify="right")

    found_input = False
    for i, dev in enumerate(all_devices):
        if dev['max_input_channels'] > 0:
            found_input = True
            name = dev['name']
            # Highlight useful devices
            if 'stereo mix' in name.lower() or 'what u hear' in name.lower() or 'loopback' in name.lower():
                name = f"[bold yellow]{name} ← USE THIS FOR SYSTEM AUDIO[/bold yellow]"
            table.add_row(str(i), name, str(dev['max_input_channels']))

    if found_input:
        console.print(table)
    else:
        console.print("[dim]  No input devices found[/dim]")

    if show_all:
        console.print("\n[bold cyan]OUTPUT DEVICES (speakers, headphones):[/bold cyan]")
        table2 = Table(box=box.ROUNDED)
        table2.add_column("Index", style="cyan", justify="right")
        table2.add_column("Name", style="green")
        table2.add_column("Channels", justify="right")

        for i, dev in enumerate(all_devices):
            if dev['max_output_channels'] > 0:
                table2.add_row(str(i), dev['name'], str(dev['max_output_channels']))
        console.print(table2)

    console.print("\n[bold yellow]To capture SYSTEM AUDIO (Discord, YouTube, etc.):[/bold yellow]")
    console.print("""
  [cyan]Windows:[/cyan]
  1. Right-click speaker icon → Sound Settings → Sound Control Panel
  2. Recording tab → Right-click → Show Disabled Devices
  3. Enable "Stereo Mix" (if available)
  4. Then use: [green]python -m src.cli record start --device INDEX[/green]
     (where INDEX is the Stereo Mix device number)

  [cyan]Alternative:[/cyan] Install VB-Audio Virtual Cable (free)
  https://vb-audio.com/Cable/

  [cyan]For microphone only:[/cyan]
  Just use any input device index shown above.
""")
    console.print("[dim]Use: python -m src.cli record start --device INDEX[/dim]")


@record.command('start')
@click.option('--device', '-d', default=None, type=int, help='Audio device index (run "record devices" to list)')
def record_start(device):
    """Start recording system audio."""
    import time
    import tempfile
    import numpy as np
    from datetime import datetime
    from .audio.audio_capture import AudioCapture, AudioChunk

    # If no device specified, list them and ask
    if device is None:
        capture = AudioCapture()
        devices = capture.list_devices()

        if devices:
            console.print("[yellow]Available audio devices:[/yellow]")
            for dev in devices:
                monitor = "[bold cyan] ← MONITOR[/bold cyan]" if 'monitor' in dev['name'].lower() else ""
                console.print(f"  [cyan]{dev['index']}[/cyan]: {dev['name']}{monitor}")
            console.print("\n[dim]Tip: Use --device INDEX to select, e.g.: record start --device 5[/dim]")
            console.print("[dim]Look for 'monitor' devices to capture system audio.[/dim]\n")

        # Try to auto-detect monitor
        for dev in devices:
            if 'monitor' in dev['name'].lower():
                device = dev['index']
                console.print(f"[green]Auto-selected monitor device: {device}[/green]\n")
                break

        if device is None and devices:
            device = devices[0]['index']
            console.print(f"[yellow]Using first available device: {device}[/yellow]\n")

        if device is None:
            console.print("[red]No audio device found.[/red]")
            return

    console.print(f"[bold green]Starting audio capture on device {device}...[/bold green]")

    # Collect audio chunks
    audio_chunks = []
    sample_rate = 16000

    def on_chunk(chunk: AudioChunk):
        audio_chunks.append(chunk.data)
        elapsed = len(audio_chunks) * 5  # ~5 sec chunks
        console.print(f"  [dim]Recording... {elapsed}s captured[/dim]", end="\r")

    try:
        capture = AudioCapture(device_name=device, sample_rate=sample_rate, callback=on_chunk)
        capture.start_recording()
        console.print("[green]Recording started. Press Ctrl+C to stop and process.[/green]")

        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        capture.stop_recording()
        console.print("\n[yellow]Recording stopped. Processing...[/yellow]")

        if not audio_chunks:
            console.print("[red]No audio captured.[/red]")
            return

        # Combine all audio
        all_audio = np.concatenate(audio_chunks)
        duration = len(all_audio) / sample_rate
        console.print(f"[cyan]Captured {duration:.1f} seconds of audio[/cyan]")

        # Save to temp file
        from scipy.io import wavfile
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_int16 = (all_audio * 32767).astype(np.int16)
        wavfile.write(temp_file.name, sample_rate, audio_int16)
        console.print(f"[dim]Saved to: {temp_file.name}[/dim]")

        # Transcribe with Whisper
        console.print("[cyan]Transcribing with Whisper...[/cyan]")
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_file.name)
            transcript = result["text"].strip()
            segments = result.get("segments", [])

            if not transcript:
                console.print("[yellow]No speech detected in recording.[/yellow]")
                return

            # Try speaker diarization
            speaker_segments = {}
            try:
                from .audio.speaker_identifier import SpeakerIdentifier, PYANNOTE_AVAILABLE
                if PYANNOTE_AVAILABLE and config.HF_TOKEN:
                    console.print("[cyan]Running speaker diarization...[/cyan]")
                    identifier = SpeakerIdentifier()
                    diarization = identifier.diarize(temp_file.name)

                    if diarization.speakers:
                        console.print(f"[green]Detected {len(diarization.speakers)} speakers: {', '.join(diarization.speakers)}[/green]")

                        # Match whisper segments to speakers
                        for seg in segments:
                            seg_mid = (seg['start'] + seg['end']) / 2
                            # Find which speaker was talking at this time
                            for d_seg in diarization.segments:
                                if d_seg.start <= seg_mid <= d_seg.end:
                                    speaker = d_seg.speaker
                                    if speaker not in speaker_segments:
                                        speaker_segments[speaker] = []
                                    speaker_segments[speaker].append(seg['text'])
                                    break
                else:
                    if not config.HF_TOKEN:
                        console.print("[yellow]No HF_TOKEN set - skipping speaker diarization[/yellow]")
                        console.print("[dim]Set HF_TOKEN in .env for multi-speaker detection[/dim]")
            except Exception as e:
                console.print(f"[yellow]Diarization skipped: {e}[/yellow]")

            # Display transcript
            if speaker_segments:
                console.print(f"\n[bold]Transcript by Speaker:[/bold]")
                for speaker, texts in speaker_segments.items():
                    console.print(f"\n[cyan]{speaker}:[/cyan]")
                    console.print(f"  {' '.join(texts)[:300]}...")
            else:
                console.print(f"\n[bold]Transcript:[/bold]\n{transcript[:500]}{'...' if len(transcript) > 500 else ''}\n")

            # Save session to database
            import uuid
            db = init_db()
            db_session = db.get_new_session()

            recording = Session(
                session_id=str(uuid.uuid4()),
                name=f"Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=int(duration),
                audio_file_path=temp_file.name,
            )
            db_session.add(recording)
            db_session.commit()

            # Create speakers and utterances
            speakers_created = {}
            if speaker_segments:
                # Multiple speakers detected
                for speaker_name, texts in speaker_segments.items():
                    speaker = Speaker(
                        speaker_id=f"{speaker_name}_{recording.id}",
                        display_name=speaker_name,
                    )
                    db_session.add(speaker)
                    db_session.commit()
                    speakers_created[speaker_name] = speaker

                    # Create utterance for this speaker
                    speaker_text = ' '.join(texts)
                    utterance = Utterance(
                        session_id=recording.id,
                        speaker_db_id=speaker.id,
                        text=speaker_text,
                        start_time=0,
                        end_time=duration,
                        word_count=len(speaker_text.split()),
                    )
                    db_session.add(utterance)
                db_session.commit()
            else:
                # Single speaker fallback
                speaker = Speaker(
                    speaker_id=f"Speaker_{recording.id}",
                    display_name="Unknown Speaker",
                )
                db_session.add(speaker)
                db_session.commit()
                speakers_created["Unknown"] = speaker

                utterance = Utterance(
                    session_id=recording.id,
                    speaker_db_id=speaker.id,
                    text=transcript,
                    start_time=0,
                    end_time=duration,
                    word_count=len(transcript.split()),
                )
                db_session.add(utterance)
                db_session.commit()

            # Get ID before closing session
            recording_id = recording.id
            db_session.close()

            console.print(f"[green]Session saved (ID: {recording_id})[/green]")

            # Profile each speaker
            analyzer = LinguisticAnalyzer(use_spacy=False)

            if speaker_segments:
                for speaker_name, texts in speaker_segments.items():
                    speaker_text = ' '.join(texts)
                    console.print(f"\n[bold cyan]═══ Profile: {speaker_name} ═══[/bold cyan]")
                    analysis = analyzer.analyze(speaker_text)
                    _display_analysis(analysis, speaker_text)
            else:
                console.print("\n[cyan]Analyzing linguistic patterns...[/cyan]")
                analysis = analyzer.analyze(transcript)
                _display_analysis(analysis, transcript)

        except ImportError:
            console.print("[yellow]Whisper not installed. Saving raw audio only.[/yellow]")
            console.print(f"Audio saved to: {temp_file.name}")
            console.print("Process later with: python -m src.cli process " + temp_file.name)

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install sounddevice")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


@record.command('stop')
def record_stop():
    """Stop recording (placeholder - recording controlled by Ctrl+C)."""
    console.print("[yellow]Recording is controlled by Ctrl+C in the start command.[/yellow]")


# =============================================================================
# Speaker Commands
# =============================================================================

@cli.group()
def speakers():
    """Manage speakers."""
    pass


@speakers.command('list')
def speakers_list():
    """List all speakers."""
    db = init_db()

    with db.get_session() as session:
        all_speakers = session.query(Speaker).order_by(Speaker.last_seen.desc()).all()

        if not all_speakers:
            console.print("[yellow]No speakers found.[/yellow]")
            return

        table = Table(title="Speakers", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Last Seen", style="yellow")
        table.add_column("Confidence", style="magenta")

        for speaker in all_speakers:
            # Get latest profile
            profile = (
                session.query(SpeakerProfile)
                .filter_by(speaker_db_id=speaker.id)
                .order_by(SpeakerProfile.updated_at.desc())
                .first()
            )

            confidence = profile.confidence_level if profile else "none"
            last_seen = speaker.last_seen.strftime("%Y-%m-%d %H:%M") if speaker.last_seen else "N/A"

            table.add_row(
                speaker.speaker_id,
                speaker.display_name or "-",
                last_seen,
                confidence
            )

        console.print(table)


@speakers.command('profile')
@click.argument('speaker_id')
def speakers_profile(speaker_id):
    """View detailed profile for a speaker."""
    db = init_db()

    with db.get_session() as session:
        profiler = BehavioralProfiler(session)
        summary = profiler.generate_profile_summary(speaker_id)

        if "error" in summary:
            console.print(f"[red]{summary['error']}[/red]")
            return

        # Display profile
        console.print(Panel(
            f"[bold]{summary['speaker']['name']}[/bold]\n"
            f"ID: {summary['speaker']['id']}\n"
            f"Last seen: {summary['speaker']['last_seen'] or 'N/A'}",
            title="Speaker Profile",
            border_style="blue"
        ))

        # Confidence and sample
        console.print(f"\n[yellow]Confidence:[/yellow] {summary['confidence']}")
        console.print(f"[yellow]Sample size:[/yellow] {summary['sample_size']} words")

        # Social Needs
        console.print("\n[bold cyan]Social Needs[/bold cyan]")
        needs = summary['dominant_needs']
        console.print(f"  Primary: [green]{needs['primary'] or 'Unknown'}[/green]")
        console.print(f"  Secondary: [green]{needs['secondary'] or 'Unknown'}[/green]")

        if needs['scores']:
            table = Table(box=box.SIMPLE)
            table.add_column("Need", style="cyan")
            table.add_column("Score", style="yellow")
            for need, score in needs['scores'].items():
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                table.add_row(need.capitalize(), f"{bar} {score:.2f}")
            console.print(table)

        # VAK
        console.print("\n[bold cyan]Sensory Modality (VAK)[/bold cyan]")
        vak = summary['vak']
        console.print(f"  Dominant: [green]{vak['dominant'].upper()}[/green]")

        for modality, score in vak['distribution'].items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            label = modality[0].upper()  # V, A, or K
            console.print(f"  {label}: {bar} {score:.2f}")

        # Decision Styles
        if summary['decision_styles']:
            console.print("\n[bold cyan]Decision Styles[/bold cyan]")
            for style in summary['decision_styles']:
                console.print(f"  • {style.capitalize()}")

        # Communication Patterns
        console.print("\n[bold cyan]Communication Patterns[/bold cyan]")
        comm = summary['communication']
        console.print(f"  Certainty level: {comm['certainty']:.2f}")
        console.print(f"  Question ratio: {comm['question_ratio']:.2f}")
        console.print(f"  Active voice: {comm['active_voice']:.2f}")
        console.print(f"  Time orientation: {comm['time_orientation']}")

        # Sentiment & Complexity
        console.print(f"\n[bold cyan]Additional Insights[/bold cyan]")
        console.print(f"  Sentiment: {summary['sentiment']['label']} ({summary['sentiment']['average']:.2f})")
        console.print(f"  Complexity: {summary['complexity']['label']} (grade {summary['complexity']['score']})")


@speakers.command('rename')
@click.argument('old_id')
@click.argument('new_name')
def speakers_rename(old_id, new_name):
    """Rename a speaker."""
    db = init_db()

    with db.get_session() as session:
        profiler = BehavioralProfiler(session)
        success = profiler.rename_speaker(old_id, new_name)

        if success:
            console.print(f"[green]Renamed {old_id} → {new_name}[/green]")
        else:
            console.print(f"[red]Speaker '{old_id}' not found.[/red]")


# =============================================================================
# Session Commands
# =============================================================================

@cli.group()
def sessions():
    """Manage recording sessions."""
    pass


@sessions.command('list')
def sessions_list():
    """List all recording sessions."""
    db = init_db()

    with db.get_session() as session:
        all_sessions = session.query(Session).order_by(Session.start_time.desc()).all()

        if not all_sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        table = Table(title="Sessions", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Date", style="yellow")
        table.add_column("Duration", style="magenta")
        table.add_column("Speakers", style="blue")
        table.add_column("Status", style="red")

        for sess in all_sessions:
            duration = f"{sess.duration_seconds // 60}m {sess.duration_seconds % 60}s" if sess.duration_seconds else "N/A"
            date = sess.start_time.strftime("%Y-%m-%d %H:%M") if sess.start_time else "N/A"

            table.add_row(
                sess.session_id[:8] + "...",
                sess.name or "-",
                date,
                duration,
                str(sess.speaker_count),
                sess.status
            )

        console.print(table)


@sessions.command('view')
@click.argument('session_id')
def sessions_view(session_id):
    """View session transcript with speaker labels."""
    db = init_db()

    with db.get_session() as session:
        # Find session (allow partial ID match)
        sess = (
            session.query(Session)
            .filter(Session.session_id.like(f"{session_id}%"))
            .first()
        )

        if not sess:
            console.print(f"[red]Session not found: {session_id}[/red]")
            return

        console.print(Panel(
            f"Session: {sess.session_id}\n"
            f"Date: {sess.start_time}\n"
            f"Duration: {sess.duration_seconds}s",
            title="Session Details"
        ))

        # Get utterances
        utterances = (
            session.query(Utterance)
            .filter_by(session_id=sess.id)
            .order_by(Utterance.start_time)
            .all()
        )

        if not utterances:
            console.print("[yellow]No transcript available.[/yellow]")
            return

        # Display transcript
        console.print("\n[bold]Transcript:[/bold]\n")
        for utt in utterances:
            speaker_name = utt.speaker.name if utt.speaker else "Unknown"
            time_str = f"[{utt.start_time:.1f}s]"
            console.print(f"[cyan]{time_str}[/cyan] [green]{speaker_name}:[/green] {utt.text}")


# =============================================================================
# Processing Commands
# =============================================================================

@cli.command('process')
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--save/--no-save', default=True, help='Save to database')
def process_audio(audio_file, save):
    """Process a pre-recorded audio file."""
    from .audio.audio_processor import AudioProcessor
    from .audio.speaker_identifier import SpeakerIdentifier

    console.print(f"[bold]Processing: {audio_file}[/bold]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # Initialize components
            task = progress.add_task("Loading models...", total=None)
            processor = AudioProcessor()
            identifier = SpeakerIdentifier()

            # Transcribe
            progress.update(task, description="Transcribing audio...")
            result = processor.transcribe(audio_file)

            if not result.has_content:
                console.print("[yellow]No speech detected in audio.[/yellow]")
                return

            console.print(f"\n[green]Transcription complete![/green]")
            console.print(f"Duration: {result.duration:.1f}s")
            console.print(f"Language: {result.language}")

            # Diarize
            progress.update(task, description="Identifying speakers...")
            try:
                diarization = identifier.diarize(audio_file)
                console.print(f"Speakers found: {len(diarization.speakers)}")
            except Exception as e:
                console.print(f"[yellow]Speaker diarization skipped: {e}[/yellow]")
                diarization = None

            progress.update(task, description="Complete!")

        # Display transcript
        console.print("\n[bold]Transcript:[/bold]")
        console.print(result.text)

        # Analyze
        if save:
            db = init_db()
            with db.get_session() as session:
                profiler = BehavioralProfiler(session)

                # Create profile for the main speaker
                speaker_id = "Speaker A"  # Default if no diarization
                profile = profiler.create_profile(speaker_id, result.text)

                console.print(f"\n[green]Profile saved for {speaker_id}[/green]")
                console.print(f"Confidence: {profile.confidence_level}")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Error processing audio: {e}[/red]")
        logger.exception("Processing error")


@cli.command('analyze')
@click.argument('text', required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='Read text from file')
def analyze_text(text, file):
    """Analyze text directly for linguistic patterns."""
    if file:
        with open(file, 'r') as f:
            text = f.read()
    elif not text:
        console.print("[yellow]Enter text (Ctrl+D to finish):[/yellow]")
        text = sys.stdin.read()

    if not text.strip():
        console.print("[red]No text provided.[/red]")
        return

    analyzer = LinguisticAnalyzer()
    result = analyzer.analyze(text)

    console.print(Panel(
        f"Word count: {result.word_count}\n"
        f"Unique words: {result.unique_words}\n"
        f"Complexity: {result.complexity:.1f}",
        title="Text Analysis"
    ))

    # VAK
    console.print("\n[bold cyan]Sensory Modality (VAK)[/bold cyan]")
    dominant_vak = result.get_dominant_vak()
    for modality, score in result.vak_scores.items():
        marker = "→" if modality == dominant_vak else " "
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        console.print(f"  {marker} {modality.capitalize()}: {bar} {score:.2f}")

    # Social Needs
    console.print("\n[bold cyan]Social Needs[/bold cyan]")
    top_needs = result.get_top_needs(3)
    for need, score in sorted(result.social_needs.items(), key=lambda x: x[1], reverse=True)[:6]:
        marker = "→" if need in top_needs else " "
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        console.print(f"  {marker} {need.capitalize()}: {bar} {score:.2f}")

    # Decision Styles
    if result.decision_styles:
        console.print("\n[bold cyan]Decision Styles Detected[/bold cyan]")
        for style in result.decision_styles:
            console.print(f"  • {style.capitalize()}")

    # Communication Patterns
    console.print("\n[bold cyan]Communication Patterns[/bold cyan]")
    comm = result.communication_patterns
    console.print(f"  Certainty: {comm['certainty']:.2f}")
    console.print(f"  Question ratio: {comm['question_ratio']:.2f}")
    console.print(f"  Time orientation: {comm['time_orientation']}")
    console.print(f"  Sentiment: {result.sentiment:.2f}")


# =============================================================================
# Export Commands
# =============================================================================

@cli.group()
def export():
    """Export data to files."""
    pass


@export.command('speaker')
@click.argument('speaker_id')
@click.option('--format', '-f', type=click.Choice(['json', 'csv', 'txt']), default='json')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_speaker(speaker_id, format, output):
    """Export speaker profile to file."""
    import json
    from datetime import datetime

    db = init_db()

    with db.get_session() as session:
        profiler = BehavioralProfiler(session)
        summary = profiler.generate_profile_summary(speaker_id)

        if "error" in summary:
            console.print(f"[red]{summary['error']}[/red]")
            return

        # Generate output filename if not specified
        if not output:
            safe_name = speaker_id.replace(" ", "_").lower()
            output = f"{safe_name}_profile.{format}"

        if format == 'json':
            with open(output, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

        elif format == 'csv':
            import csv
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Category', 'Attribute', 'Value'])

                # Speaker info
                writer.writerow(['Speaker', 'ID', summary['speaker']['id']])
                writer.writerow(['Speaker', 'Name', summary['speaker']['name']])

                # Social needs
                for need, score in summary['dominant_needs']['scores'].items():
                    writer.writerow(['Social Need', need, f"{score:.3f}"])

                # VAK
                for modality, score in summary['vak']['distribution'].items():
                    writer.writerow(['VAK', modality, f"{score:.3f}"])

                # Decision styles
                for style in summary.get('decision_styles', []):
                    writer.writerow(['Decision Style', style, 'detected'])

                # Communication
                writer.writerow(['Communication', 'Certainty', f"{summary['communication']['certainty']:.3f}"])
                writer.writerow(['Communication', 'Time Orientation', summary['communication']['time_orientation']])

                # Metadata
                writer.writerow(['Metadata', 'Confidence', summary['confidence']])
                writer.writerow(['Metadata', 'Sample Size', summary['sample_size']])

        elif format == 'txt':
            with open(output, 'w') as f:
                f.write(f"SPEAKER PROFILE: {summary['speaker']['name']}\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"ID: {summary['speaker']['id']}\n")
                f.write(f"Confidence: {summary['confidence']}\n")
                f.write(f"Sample Size: {summary['sample_size']} words\n\n")

                f.write("SOCIAL NEEDS\n")
                f.write("-"*30 + "\n")
                f.write(f"Primary: {summary['dominant_needs']['primary']}\n")
                f.write(f"Secondary: {summary['dominant_needs']['secondary']}\n")
                for need, score in summary['dominant_needs']['scores'].items():
                    f.write(f"  {need}: {score:.3f}\n")

                f.write("\nSENSORY MODALITY (VAK)\n")
                f.write("-"*30 + "\n")
                f.write(f"Dominant: {summary['vak']['dominant']}\n")
                for modality, score in summary['vak']['distribution'].items():
                    f.write(f"  {modality}: {score:.3f}\n")

                if summary.get('decision_styles'):
                    f.write("\nDECISION STYLES\n")
                    f.write("-"*30 + "\n")
                    for style in summary['decision_styles']:
                        f.write(f"  - {style}\n")

                f.write("\nCOMMUNICATION PATTERNS\n")
                f.write("-"*30 + "\n")
                f.write(f"Certainty: {summary['communication']['certainty']:.3f}\n")
                f.write(f"Question Ratio: {summary['communication']['question_ratio']:.3f}\n")
                f.write(f"Active Voice: {summary['communication']['active_voice']:.3f}\n")
                f.write(f"Time Orientation: {summary['communication']['time_orientation']}\n")

                f.write(f"\nSENTIMENT: {summary['sentiment']['label']} ({summary['sentiment']['average']:.3f})\n")
                f.write(f"COMPLEXITY: {summary['complexity']['label']} (grade {summary['complexity']['score']})\n")

        console.print(f"[green]Profile exported to {output}[/green]")


@export.command('session')
@click.argument('session_id')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'csv']), default='txt')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_session(session_id, format, output):
    """Export session transcript to file."""
    import json

    db = init_db()

    with db.get_session() as session:
        # Find session
        sess = (
            session.query(Session)
            .filter(Session.session_id.like(f"{session_id}%"))
            .first()
        )

        if not sess:
            console.print(f"[red]Session not found: {session_id}[/red]")
            return

        # Get utterances
        utterances = (
            session.query(Utterance)
            .filter_by(session_id=sess.id)
            .order_by(Utterance.start_time)
            .all()
        )

        if not output:
            output = f"session_{session_id[:8]}.{format}"

        if format == 'json':
            data = {
                "session_id": sess.session_id,
                "name": sess.name,
                "start_time": sess.start_time.isoformat() if sess.start_time else None,
                "duration_seconds": sess.duration_seconds,
                "utterances": [
                    {
                        "speaker": u.speaker.name if u.speaker else "Unknown",
                        "text": u.text,
                        "start_time": u.start_time,
                        "end_time": u.end_time,
                        "word_count": u.word_count,
                        "vak": u.vak_detected,
                        "need": u.dominant_need,
                        "sentiment": u.sentiment,
                    }
                    for u in utterances
                ]
            }
            with open(output, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'txt':
            with open(output, 'w') as f:
                f.write(f"SESSION TRANSCRIPT\n")
                f.write(f"{'='*50}\n")
                f.write(f"Session ID: {sess.session_id}\n")
                f.write(f"Name: {sess.name or 'Unnamed'}\n")
                f.write(f"Date: {sess.start_time}\n")
                f.write(f"Duration: {sess.duration_seconds}s\n")
                f.write(f"{'='*50}\n\n")

                for u in utterances:
                    speaker = u.speaker.name if u.speaker else "Unknown"
                    f.write(f"[{u.start_time:.1f}s] {speaker}:\n")
                    f.write(f"  {u.text}\n\n")

        elif format == 'csv':
            import csv
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Speaker', 'Text', 'Words', 'VAK', 'Need', 'Sentiment'])
                for u in utterances:
                    speaker = u.speaker.name if u.speaker else "Unknown"
                    writer.writerow([
                        f"{u.start_time:.1f}",
                        speaker,
                        u.text,
                        u.word_count,
                        u.vak_detected or "",
                        u.dominant_need or "",
                        f"{u.sentiment:.2f}" if u.sentiment else ""
                    ])

        console.print(f"[green]Session exported to {output}[/green]")


@export.command('analysis')
@click.argument('text', required=False)
@click.option('--file', '-f', type=click.Path(exists=True), help='Read text from file')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', type=click.Choice(['json', 'txt']), default='json')
def export_analysis(text, file, output, format):
    """Analyze text and export results to file."""
    import json

    if file:
        with open(file, 'r') as f:
            text = f.read()
    elif not text:
        console.print("[yellow]Enter text (Ctrl+D to finish):[/yellow]")
        text = sys.stdin.read()

    if not text.strip():
        console.print("[red]No text provided.[/red]")
        return

    analyzer = LinguisticAnalyzer()
    result = analyzer.analyze(text)

    analysis = {
        "word_count": result.word_count,
        "unique_words": result.unique_words,
        "complexity": result.complexity,
        "sentiment": result.sentiment,
        "vak": {
            "scores": result.vak_scores,
            "dominant": result.get_dominant_vak(),
        },
        "social_needs": {
            "scores": result.social_needs,
            "top_needs": result.get_top_needs(2),
        },
        "decision_styles": result.decision_styles,
        "communication": result.communication_patterns,
        "pronouns": result.pronoun_ratios,
        "emotions": result.emotional_indicators,
        "influence_patterns": result.influence_patterns,
        "stress_indicators": result.stress_indicators,
        "filler_ratio": result.filler_ratio,
    }

    if not output:
        output = f"analysis.{format}"

    if format == 'json':
        with open(output, 'w') as f:
            json.dump(analysis, f, indent=2)
    else:
        with open(output, 'w') as f:
            f.write("LINGUISTIC ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Word Count: {analysis['word_count']}\n")
            f.write(f"Unique Words: {analysis['unique_words']}\n")
            f.write(f"Complexity (Grade): {analysis['complexity']:.1f}\n")
            f.write(f"Sentiment: {analysis['sentiment']:.3f}\n\n")

            f.write("VAK MODALITY\n" + "-"*30 + "\n")
            f.write(f"Dominant: {analysis['vak']['dominant']}\n")
            for m, s in analysis['vak']['scores'].items():
                f.write(f"  {m}: {s:.3f}\n")

            f.write("\nSOCIAL NEEDS\n" + "-"*30 + "\n")
            f.write(f"Top Needs: {', '.join(analysis['social_needs']['top_needs'])}\n")
            for n, s in sorted(analysis['social_needs']['scores'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {n}: {s:.3f}\n")

            if analysis['decision_styles']:
                f.write("\nDECISION STYLES\n" + "-"*30 + "\n")
                for style in analysis['decision_styles']:
                    f.write(f"  - {style}\n")

            f.write("\nCOMMUNICATION\n" + "-"*30 + "\n")
            for k, v in analysis['communication'].items():
                f.write(f"  {k}: {v}\n")

            if analysis['influence_patterns']:
                f.write("\nINFLUENCE PATTERNS DETECTED\n" + "-"*30 + "\n")
                for pattern in analysis['influence_patterns']:
                    f.write(f"  - {pattern}\n")

    console.print(f"[green]Analysis exported to {output}[/green]")


# =============================================================================
# Database Commands
# =============================================================================

@cli.command('init-db')
def init_database():
    """Initialize the database."""
    console.print("[yellow]Initializing database...[/yellow]")
    db = init_db()
    console.print(f"[green]Database initialized at {config.DATABASE_URL}[/green]")


@cli.command('dump')
@click.option('--output', '-o', type=click.Path(), default='profiler_dump.txt', help='Output file')
def dump_database(output):
    """Dump all sessions and transcripts to a text file."""
    db = init_db()

    with db.get_session() as session:
        all_sessions = session.query(Session).order_by(Session.start_time).all()

        if not all_sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return

        with open(output, 'w', encoding='utf-8') as f:
            f.write("PROFILER DATABASE DUMP\n")
            f.write("=" * 60 + "\n\n")

            for sess in all_sessions:
                f.write(f"SESSION: {sess.name or 'Unnamed'}\n")
                f.write(f"ID: {sess.session_id}\n")
                f.write(f"Date: {sess.start_time}\n")
                f.write(f"Duration: {sess.duration_seconds}s\n")
                f.write("-" * 40 + "\n")

                # Get utterances
                utterances = (
                    session.query(Utterance)
                    .filter_by(session_id=sess.id)
                    .order_by(Utterance.start_time)
                    .all()
                )

                for utt in utterances:
                    speaker_name = utt.speaker.display_name if utt.speaker else "Unknown"
                    f.write(f"[{speaker_name}]: {utt.text}\n")

                f.write("\n" + "=" * 60 + "\n\n")

        console.print(f"[green]Dumped {len(all_sessions)} sessions to {output}[/green]")


@speakers.command('register')
@click.argument('name')
@click.option('--file', '-f', type=click.Path(exists=True), help='Audio file to extract voice from')
@click.option('--session', '-s', type=int, help='Session ID to extract voice from')
@click.option('--device', '-d', type=int, help='Record live from device')
@click.option('--duration', default=5, help='Recording duration in seconds')
def speakers_register(name, file, session, device, duration):
    """Register a speaker's voice for real-time identification."""
    from .audio.speaker_identifier import SpeakerIdentifier, PYANNOTE_AVAILABLE
    import numpy as np

    if not PYANNOTE_AVAILABLE:
        console.print("[red]pyannote.audio required for voice registration[/red]")
        return

    db = init_db()
    identifier = SpeakerIdentifier()

    embedding = None

    if file:
        # Extract from audio file
        console.print(f"[cyan]Extracting voice from {file}...[/cyan]")
        embedding = identifier.extract_embedding(file)

    elif session:
        # Extract from session audio
        with db.get_session() as sess:
            from .storage.models import Session as SessionModel
            sess_record = sess.query(SessionModel).filter_by(id=session).first()
            if not sess_record or not sess_record.audio_file_path:
                console.print(f"[red]Session {session} not found or has no audio[/red]")
                return

            audio_path = sess_record.audio_file_path
            if not Path(audio_path).exists():
                console.print(f"[red]Audio file not found: {audio_path}[/red]")
                return

            console.print(f"[cyan]Extracting voice from session {session}...[/cyan]")
            embedding = identifier.extract_embedding(audio_path)

    elif device is not None:
        # Record live
        console.print(f"[cyan]Recording {duration} seconds from device {device}...[/cyan]")
        console.print("[yellow]Speak now![/yellow]")

        from .audio.audio_capture import AudioCapture
        import time

        chunks = []
        def on_chunk(chunk):
            chunks.append(chunk.data)

        capture = AudioCapture(device_name=device, callback=on_chunk)
        capture.start_recording()
        time.sleep(duration)
        capture.stop_recording()

        if chunks:
            audio_data = np.concatenate(chunks)
            embedding = identifier.extract_embedding(audio_data, capture.sample_rate)
    else:
        console.print("[yellow]Specify --file, --session, or --device[/yellow]")
        console.print("Examples:")
        console.print("  speakers register John --file recording.wav")
        console.print("  speakers register John --session 13")
        console.print("  speakers register John --device 27")
        return

    if embedding is None:
        console.print("[red]Failed to extract voice embedding[/red]")
        return

    # Save embedding to database
    with db.get_session() as sess:
        # Check if speaker exists
        speaker = sess.query(Speaker).filter_by(display_name=name).first()
        if not speaker:
            speaker = Speaker(
                speaker_id=f"registered_{name.lower().replace(' ', '_')}",
                display_name=name,
            )
            sess.add(speaker)
            sess.commit()

        # Store embedding as JSON in metadata or separate table
        # For now, store in the identifier's memory and save to file
        identifier.register_speaker(embedding, name)

        # Save embeddings to file
        embeddings_file = Path(DATA_DIR) / "voice_embeddings.json"
        import json
        existing = {}
        if embeddings_file.exists():
            with open(embeddings_file) as f:
                existing = json.load(f)

        existing[name] = embedding.tolist()
        with open(embeddings_file, 'w') as f:
            json.dump(existing, f)

        console.print(f"[green]Registered voice for: {name}[/green]")
        console.print(f"[dim]Saved to {embeddings_file}[/dim]")


@speakers.command('voices')
def speakers_voices():
    """List all registered voice embeddings."""
    import json

    embeddings_file = Path(DATA_DIR) / "voice_embeddings.json"
    if not embeddings_file.exists():
        console.print("[yellow]No voices registered yet.[/yellow]")
        console.print("Use: speakers register <name> --device 27")
        return

    with open(embeddings_file) as f:
        embeddings = json.load(f)

    console.print(f"[bold]Registered Voices ({len(embeddings)}):[/bold]")
    for name in embeddings:
        console.print(f"  • {name}")


@sessions.command('audio')
def sessions_audio():
    """List sessions that have audio files."""
    db = init_db()

    with db.get_session() as session:
        all_sessions = session.query(Session).filter(
            Session.audio_file_path.isnot(None)
        ).order_by(Session.start_time.desc()).all()

        if not all_sessions:
            console.print("[yellow]No sessions with audio files.[/yellow]")
            return

        table = Table(title="Sessions with Audio", box=box.ROUNDED)
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Date", style="yellow")
        table.add_column("Audio File", style="dim")
        table.add_column("Exists", style="magenta")

        for sess in all_sessions:
            exists = "✓" if Path(sess.audio_file_path).exists() else "✗"
            audio_short = sess.audio_file_path[-40:] if sess.audio_file_path else "-"
            date = sess.start_time.strftime("%Y-%m-%d %H:%M") if sess.start_time else "N/A"

            table.add_row(
                str(sess.id),
                sess.name or "-",
                date,
                f"...{audio_short}",
                exists
            )

        console.print(table)
        console.print("\n[dim]Use: speakers register <name> --session <ID>[/dim]")


@speakers.command('rename-interactive')
def speakers_rename_interactive():
    """Interactively rename all speakers."""
    db = init_db()

    with db.get_session() as session:
        all_speakers = session.query(Speaker).order_by(Speaker.last_seen.desc()).all()

        if not all_speakers:
            console.print("[yellow]No speakers found.[/yellow]")
            return

        console.print("[bold]Rename speakers (press Enter to skip, 'q' to quit)[/bold]\n")

        for speaker in all_speakers:
            # Show speaker info
            console.print(f"[cyan]Current: {speaker.display_name or speaker.speaker_id}[/cyan]")
            console.print(f"  ID: {speaker.speaker_id}")
            console.print(f"  Last seen: {speaker.last_seen}")

            # Get sample text
            sample = session.query(Utterance).filter_by(speaker_db_id=speaker.id).first()
            if sample:
                preview = sample.text[:100] + "..." if len(sample.text) > 100 else sample.text
                console.print(f"  Sample: [dim]{preview}[/dim]")

            new_name = console.input("[yellow]New name: [/yellow]").strip()

            if new_name.lower() == 'q':
                console.print("[yellow]Cancelled.[/yellow]")
                break
            elif new_name:
                speaker.display_name = new_name
                session.commit()
                console.print(f"[green]Renamed to: {new_name}[/green]")

            console.print()

        console.print("[green]Done![/green]")


@speakers.command('all')
def speakers_all():
    """List all speakers with details and sample text."""
    db = init_db()

    with db.get_session() as session:
        all_speakers = session.query(Speaker).order_by(Speaker.display_name, Speaker.last_seen.desc()).all()

        if not all_speakers:
            console.print("[yellow]No speakers found.[/yellow]")
            return

        console.print(f"[bold]All Speakers ({len(all_speakers)}):[/bold]\n")

        for speaker in all_speakers:
            # Get all utterances for word count
            utterances = session.query(Utterance).filter_by(speaker_db_id=speaker.id).all()
            total_words = sum(u.word_count or 0 for u in utterances)
            session_ids = set(u.session_id for u in utterances)

            # Get sample text
            sample_text = ""
            if utterances:
                sample_text = utterances[0].text[:80] + "..." if len(utterances[0].text) > 80 else utterances[0].text

            # Display
            name = speaker.display_name or speaker.speaker_id
            console.print(f"[bold cyan]#{speaker.id}[/bold cyan] [green]{name}[/green]")
            console.print(f"    Words: {total_words} | Sessions: {len(session_ids)} | Last: {speaker.last_seen or 'N/A'}")
            if sample_text:
                console.print(f"    [dim]\"{sample_text}\"[/dim]")
            console.print()


@speakers.command('merge')
@click.argument('source_id', type=int)
@click.argument('target_id', type=int)
def speakers_merge(source_id, target_id):
    """Merge two speakers - moves all utterances from source to target.

    Example: speakers merge 5 3
    (Moves all data from speaker #5 into speaker #3, then deletes #5)
    """
    db = init_db()

    with db.get_session() as session:
        source = session.query(Speaker).filter_by(id=source_id).first()
        target = session.query(Speaker).filter_by(id=target_id).first()

        if not source:
            console.print(f"[red]Source speaker #{source_id} not found[/red]")
            return
        if not target:
            console.print(f"[red]Target speaker #{target_id} not found[/red]")
            return

        source_name = source.display_name or source.speaker_id
        target_name = target.display_name or target.speaker_id

        # Count utterances
        source_utterances = session.query(Utterance).filter_by(speaker_db_id=source.id).all()

        console.print(f"[yellow]Merging:[/yellow]")
        console.print(f"  Source: #{source_id} {source_name} ({len(source_utterances)} utterances)")
        console.print(f"  Target: #{target_id} {target_name}")

        confirm = console.input("\n[yellow]Proceed? (y/n): [/yellow]").strip().lower()
        if confirm != 'y':
            console.print("[yellow]Cancelled.[/yellow]")
            return

        # Move all utterances
        for utt in source_utterances:
            utt.speaker_db_id = target.id

        # Delete source speaker
        session.delete(source)
        session.commit()

        console.print(f"[green]Merged {len(source_utterances)} utterances into {target_name}[/green]")
        console.print(f"[green]Deleted speaker #{source_id}[/green]")


@speakers.command('manage')
def speakers_manage():
    """Interactive speaker management - view, rename, merge."""
    db = init_db()

    while True:
        with db.get_session() as session:
            all_speakers = session.query(Speaker).order_by(Speaker.display_name, Speaker.id).all()

            if not all_speakers:
                console.print("[yellow]No speakers found.[/yellow]")
                return

            # Display all speakers
            console.print("\n[bold]═══ Speaker Management ═══[/bold]\n")

            for speaker in all_speakers:
                utterances = session.query(Utterance).filter_by(speaker_db_id=speaker.id).all()
                total_words = sum(u.word_count or 0 for u in utterances)
                session_count = len(set(u.session_id for u in utterances))

                name = speaker.display_name or speaker.speaker_id
                sample = ""
                if utterances:
                    sample = utterances[0].text[:60] + "..." if len(utterances[0].text) > 60 else utterances[0].text

                console.print(f"[cyan]#{speaker.id:3}[/cyan] [green]{name:20}[/green] {total_words:5} words | {session_count} sessions")
                if sample:
                    console.print(f"      [dim]\"{sample}\"[/dim]")

        console.print("\n[bold]Commands:[/bold]")
        console.print("  [cyan]r <id> <name>[/cyan]  - Rename speaker")
        console.print("  [cyan]m <src> <tgt>[/cyan]  - Merge speakers (move src → tgt)")
        console.print("  [cyan]d <id>[/cyan]         - Delete speaker")
        console.print("  [cyan]v <id>[/cyan]         - View speaker's full text")
        console.print("  [cyan]q[/cyan]              - Quit")

        cmd = console.input("\n[yellow]> [/yellow]").strip()

        if not cmd or cmd.lower() == 'q':
            break

        parts = cmd.split(maxsplit=2)
        action = parts[0].lower()

        try:
            with db.get_session() as session:
                if action == 'r' and len(parts) >= 3:
                    # Rename
                    speaker_id = int(parts[1])
                    new_name = parts[2]
                    speaker = session.query(Speaker).filter_by(id=speaker_id).first()
                    if speaker:
                        old_name = speaker.display_name or speaker.speaker_id
                        speaker.display_name = new_name
                        session.commit()
                        console.print(f"[green]Renamed: {old_name} → {new_name}[/green]")
                    else:
                        console.print(f"[red]Speaker #{speaker_id} not found[/red]")

                elif action == 'm' and len(parts) >= 3:
                    # Merge
                    src_id = int(parts[1])
                    tgt_id = int(parts[2])
                    source = session.query(Speaker).filter_by(id=src_id).first()
                    target = session.query(Speaker).filter_by(id=tgt_id).first()

                    if not source or not target:
                        console.print("[red]Invalid speaker IDs[/red]")
                    else:
                        # Move utterances
                        count = session.query(Utterance).filter_by(speaker_db_id=src_id).update(
                            {Utterance.speaker_db_id: tgt_id}
                        )
                        session.delete(source)
                        session.commit()
                        console.print(f"[green]Merged {count} utterances from #{src_id} → #{tgt_id}[/green]")

                elif action == 'd' and len(parts) >= 2:
                    # Delete
                    speaker_id = int(parts[1])
                    speaker = session.query(Speaker).filter_by(id=speaker_id).first()
                    if speaker:
                        name = speaker.display_name or speaker.speaker_id
                        # Delete utterances first
                        session.query(Utterance).filter_by(speaker_db_id=speaker_id).delete()
                        session.delete(speaker)
                        session.commit()
                        console.print(f"[green]Deleted: {name}[/green]")
                    else:
                        console.print(f"[red]Speaker #{speaker_id} not found[/red]")

                elif action == 'v' and len(parts) >= 2:
                    # View full text
                    speaker_id = int(parts[1])
                    speaker = session.query(Speaker).filter_by(id=speaker_id).first()
                    if speaker:
                        utterances = session.query(Utterance).filter_by(speaker_db_id=speaker_id).all()
                        name = speaker.display_name or speaker.speaker_id
                        console.print(f"\n[bold cyan]═══ {name} ═══[/bold cyan]\n")
                        for utt in utterances:
                            console.print(f"[dim]Session {utt.session_id}:[/dim] {utt.text}\n")
                        console.input("[dim]Press Enter to continue...[/dim]")
                    else:
                        console.print(f"[red]Speaker #{speaker_id} not found[/red]")

                else:
                    console.print("[red]Invalid command. Try: r/m/d/v/q[/red]")

        except ValueError:
            console.print("[red]Invalid ID - must be a number[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[green]Done![/green]")


@cli.command('overlay')
@click.option('--device', '-d', type=int, required=True, help='Audio device index')
def start_overlay(device):
    """Start real-time profiling overlay."""
    from .overlay import run_overlay, TK_AVAILABLE

    if not TK_AVAILABLE:
        console.print("[red]tkinter not available. Install with: pip install tk[/red]")
        return

    console.print(f"[cyan]Starting overlay on device {device}...[/cyan]")
    console.print("[dim]Drag to move, click X to close[/dim]")

    try:
        run_overlay(device)
    except KeyboardInterrupt:
        console.print("\n[yellow]Overlay closed.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command('stats')
def show_stats():
    """Show system statistics."""
    db = init_db()

    with db.get_session() as session:
        speaker_count = session.query(Speaker).count()
        session_count = session.query(Session).count()
        utterance_count = session.query(Utterance).count()

        # Total words
        total_words = session.query(
            Utterance
        ).with_entities(
            Utterance.word_count
        ).all()
        total_word_count = sum(u[0] or 0 for u in total_words)

        console.print(Panel(
            f"Speakers: {speaker_count}\n"
            f"Sessions: {session_count}\n"
            f"Utterances: {utterance_count}\n"
            f"Total words: {total_word_count:,}",
            title="System Statistics"
        ))


@cli.command('compare')
@click.argument('speaker1')
@click.argument('speaker2')
def compare_profiles(speaker1, speaker2):
    """Compare two speaker profiles side by side."""
    db = init_db()

    with db.get_session() as session:
        profiler = BehavioralProfiler(session)
        summary1 = profiler.generate_profile_summary(speaker1)
        summary2 = profiler.generate_profile_summary(speaker2)

        if "error" in summary1:
            console.print(f"[red]{summary1['error']}[/red]")
            return
        if "error" in summary2:
            console.print(f"[red]{summary2['error']}[/red]")
            return

        # Header
        console.print(Panel(
            f"Comparing: [cyan]{summary1['speaker']['name']}[/cyan] vs [yellow]{summary2['speaker']['name']}[/yellow]",
            title="Profile Comparison"
        ))

        # Social Needs comparison
        console.print("\n[bold]Social Needs[/bold]")
        table = Table(box=box.SIMPLE)
        table.add_column("Need", style="cyan")
        table.add_column(summary1['speaker']['name'][:15], justify="right")
        table.add_column("", justify="center")
        table.add_column(summary2['speaker']['name'][:15], justify="left")

        needs1 = summary1['dominant_needs']['scores']
        needs2 = summary2['dominant_needs']['scores']

        for need in ['significance', 'approval', 'acceptance', 'intelligence', 'pity', 'power']:
            s1 = needs1.get(need, 0)
            s2 = needs2.get(need, 0)
            bar1 = "█" * int(s1 * 10)
            bar2 = "█" * int(s2 * 10)
            diff = "=" if abs(s1 - s2) < 0.1 else (">" if s1 > s2 else "<")
            table.add_row(
                need.capitalize(),
                f"{bar1} {s1:.2f}",
                diff,
                f"{s2:.2f} {bar2}"
            )

        console.print(table)

        # VAK comparison
        console.print("\n[bold]Sensory Modality (VAK)[/bold]")
        vak1 = summary1['vak']['distribution']
        vak2 = summary2['vak']['distribution']

        for modality in ['visual', 'auditory', 'kinesthetic']:
            s1 = vak1.get(modality, 0)
            s2 = vak2.get(modality, 0)
            console.print(f"  {modality[0].upper()}: {s1:.2f} {'←' if s1 > s2 else '→' if s2 > s1 else '='} {s2:.2f}")

        console.print(f"\n  Dominant: [cyan]{summary1['vak']['dominant']}[/cyan] vs [yellow]{summary2['vak']['dominant']}[/yellow]")

        # Decision styles
        styles1 = set(summary1.get('decision_styles', []))
        styles2 = set(summary2.get('decision_styles', []))

        console.print("\n[bold]Decision Styles[/bold]")
        shared = styles1 & styles2
        only1 = styles1 - styles2
        only2 = styles2 - styles1

        if shared:
            console.print(f"  Shared: {', '.join(shared)}")
        if only1:
            console.print(f"  Only {summary1['speaker']['name']}: {', '.join(only1)}")
        if only2:
            console.print(f"  Only {summary2['speaker']['name']}: {', '.join(only2)}")

        # Communication style
        console.print("\n[bold]Communication Style[/bold]")
        comm1 = summary1['communication']
        comm2 = summary2['communication']

        console.print(f"  Certainty: {comm1['certainty']:.2f} vs {comm2['certainty']:.2f}")
        console.print(f"  Time Focus: {comm1['time_orientation']} vs {comm2['time_orientation']}")

        # Compatibility assessment
        console.print("\n[bold]Rapport Suggestions[/bold]")

        # VAK matching tip
        dom_vak1 = summary1['vak']['dominant']
        dom_vak2 = summary2['vak']['dominant']
        if dom_vak1 == dom_vak2:
            console.print(f"  [green]✓[/green] Both are {dom_vak1} - use {dom_vak1} language")
        else:
            console.print(f"  [yellow]![/yellow] Different modalities - {summary1['speaker']['name']} prefers {dom_vak1}, {summary2['speaker']['name']} prefers {dom_vak2}")

        # Social needs tip
        top1 = summary1['dominant_needs']['primary']
        top2 = summary2['dominant_needs']['primary']
        if top1 == top2:
            console.print(f"  [green]✓[/green] Both driven by {top1}")
        else:
            console.print(f"  [yellow]![/yellow] {summary1['speaker']['name']} needs {top1}, {summary2['speaker']['name']} needs {top2}")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    setup_logging()
    cli()


if __name__ == '__main__':
    main()
