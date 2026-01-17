"""Real-time overlay for speaker profiling."""

import json
import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np

# Claude analysis thresholds
CLAUDE_ANALYSIS_THRESHOLD = 200  # Words before first analysis
CLAUDE_REANALYSIS_INTERVAL = 200  # Words between re-analyses

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

from .config import config, DATA_DIR
from .audio.audio_capture import AudioCapture
from .audio.speaker_identifier import SpeakerIdentifier, PYANNOTE_AVAILABLE
from .profiling.linguistic_analyzer import LinguisticAnalyzer


# Communication tips based on profile
VAK_TIPS = {
    "visual": [
        "Use words like: see, look, picture, imagine, view",
        "Show diagrams, charts, or written info",
        "Say: 'Let me show you...' or 'Picture this...'",
    ],
    "auditory": [
        "Use words like: hear, sounds, listen, tell, discuss",
        "Explain things verbally, use tone variety",
        "Say: 'How does that sound?' or 'Let me tell you...'",
    ],
    "kinesthetic": [
        "Use words like: feel, touch, grasp, handle, concrete",
        "Let them try things hands-on",
        "Say: 'How do you feel about...' or 'Let's walk through...'",
    ],
}

NEED_TIPS = {
    "significance": [
        "Acknowledge their achievements",
        "Ask about their unique contributions",
        "Highlight how they stand out",
    ],
    "approval": [
        "Give genuine compliments",
        "Show appreciation for their efforts",
        "Be supportive and encouraging",
    ],
    "acceptance": [
        "Emphasize group belonging",
        "Use inclusive language (we, us, together)",
        "Make them feel part of the team",
    ],
    "intelligence": [
        "Respect their expertise",
        "Ask for their analysis/opinion",
        "Present logical arguments",
    ],
    "power": [
        "Give them choices and control",
        "Ask what they want to do",
        "Let them lead when possible",
    ],
    "pity": [
        "Show empathy and understanding",
        "Acknowledge their struggles",
        "Offer support without judgment",
    ],
}


class RealtimeProfiler:
    """Real-time audio profiling with speaker matching."""

    def __init__(self, device: int):
        self.device = device
        self.running = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Load registered voice embeddings
        self.known_voices: Dict[str, np.ndarray] = {}
        self._load_voice_embeddings()

        # Track unknown speakers this session (auto-clustering)
        self.session_voices: Dict[str, list] = {}  # "Unknown 1" -> [embeddings list]
        self.unknown_counter = 0
        self.similarity_threshold = 0.45  # Lower threshold for more lenient matching

        # Initialize components
        self.identifier = SpeakerIdentifier() if PYANNOTE_AVAILABLE else None
        self.analyzer = LinguisticAnalyzer(use_spacy=False)

        # Current state
        self.current_speaker = "Unknown"
        self.current_profile = {}
        self.speaker_profiles: Dict[str, dict] = {}

        # Claude analysis tracking
        self.last_claude_word_count: Dict[str, int] = {}  # speaker -> word count at last analysis
        self.claude_insights: Dict[str, dict] = {}  # speaker -> latest Claude insights

    def _load_voice_embeddings(self):
        """Load registered voice embeddings."""
        embeddings_file = Path(DATA_DIR) / "voice_embeddings.json"
        if embeddings_file.exists():
            with open(embeddings_file) as f:
                data = json.load(f)
                self.known_voices = {
                    name: np.array(emb) for name, emb in data.items()
                }

    def _match_speaker(self, embedding: np.ndarray) -> tuple:
        """Match embedding to known or session speakers."""
        from scipy.spatial.distance import cosine

        best_match = None
        best_score = 0.0

        # First check registered voices
        for name, known_emb in self.known_voices.items():
            similarity = 1 - cosine(embedding, known_emb)
            if similarity > best_score and similarity > self.similarity_threshold:
                best_match = name
                best_score = similarity

        # If no registered match, check session voices (unknown speakers)
        if best_match is None:
            for name, emb_list in self.session_voices.items():
                # Compare against average embedding for stability
                avg_emb = np.mean(emb_list, axis=0)
                similarity = 1 - cosine(embedding, avg_emb)
                if similarity > best_score and similarity > self.similarity_threshold:
                    best_match = name
                    best_score = similarity

        # If match found in session, add to their embedding list
        if best_match and best_match in self.session_voices:
            self.session_voices[best_match].append(embedding)
            # Keep only last 10 embeddings
            if len(self.session_voices[best_match]) > 10:
                self.session_voices[best_match] = self.session_voices[best_match][-10:]

        # If still no match, create new unknown speaker
        if best_match is None:
            self.unknown_counter += 1
            best_match = f"Unknown {self.unknown_counter}"
            self.session_voices[best_match] = [embedding]
            best_score = 1.0  # Perfect match to self

        return best_match, best_score

    def _analyze_with_claude(self, speaker: str, text: str) -> Optional[dict]:
        """Send text to Claude for deeper personality analysis."""
        try:
            import anthropic
        except ImportError:
            print("[Claude] anthropic package not installed. Run: pip install anthropic")
            return None

        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            print("[Claude] ANTHROPIC_API_KEY not set in config")
            return None

        try:
            client = anthropic.Anthropic(api_key=api_key)

            # Get deception markers if available for context
            deception_context = ""
            if speaker in self.speaker_profiles:
                prof = self.speaker_profiles[speaker]
                pol = prof.get("politician_score", 0)
                dec = prof.get("deception_score", 0)
                if pol > 0.2 or dec > 0.2:
                    deception_context = f"\n\nNOTE: Our linguistic analysis detected politician_score={pol:.0%}, deception_score={dec:.0%}"

            prompt = f"""Analyze this person's speech patterns and provide insights about their personality, communication style, and DECEPTION MARKERS.

Speaker: {speaker}
Text sample ({len(text.split())} words):
"{text}"
{deception_context}

DECEPTION PATTERNS TO LOOK FOR:
1. FALSE EMPATHY - Rich/powerful people saying "I feel your pain", "you're not alone", "I know how hard it is" (performative concern)
2. FALSE RELATABILITY - "working families", "kitchen table", "putting food on the table" (millionaires pretending to be regular folks)
3. BLAME SHIFTING - Mentioning other politicians/parties to deflect ("under [opponent]", "the previous administration")
4. HEDGING - "I believe", "to my knowledge", "I don't recall" (avoiding commitment)
5. NON-ANSWERS - "That's a great question", "Let me be clear" then not being clear
6. WEASEL WORDS - "Some people say", "Many believe", "Studies show" (vague attribution)
7. STATS AS MANIPULATION - Cherry-picked statistics to seem authoritative
8. FAKE NICENESS - "With all due respect", "My good friend" (saccharine politeness)
9. FUTURE FAKING - "We're looking into it", "Very soon" (vague promises)
10. EMOTIONAL MANIPULATION - "Think of the children", "Our freedom" (appeals over substance)

Based on this speech sample, provide a JSON response with:
{{
    "personality_summary": "2-3 sentence summary of their personality",
    "communication_style": "direct/indirect, formal/casual, etc.",
    "likely_values": ["3-5", "core", "values"],
    "how_to_persuade": "Best approach to influence this person",
    "rapport_tip": "One specific tip to build rapport right now",
    "honesty_assessment": "honest/evasive/manipulative - BE HARSH, call out BS",
    "deception_detected": ["list", "specific", "deception", "patterns", "found"],
    "specific_red_flags": "Quote specific phrases that are manipulative or deceptive"
}}

BE CRITICAL. If this sounds like a politician pandering, SAY SO. If they're using false empathy or relatability, CALL IT OUT.

Respond ONLY with the JSON object, no other text."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text.strip()

            # Parse JSON from response
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            insights = json.loads(response_text)
            print(f"[Claude] Analysis complete for {speaker}")
            return insights

        except json.JSONDecodeError as e:
            print(f"[Claude] Failed to parse response: {e}")
            return None
        except Exception as e:
            print(f"[Claude] API error: {e}")
            return None

    def _maybe_trigger_claude_analysis(self, speaker: str):
        """Check if we should trigger Claude analysis for this speaker."""
        if speaker not in self.speaker_profiles:
            return

        profile = self.speaker_profiles[speaker]
        word_count = profile.get("word_count", 0)
        last_analyzed = self.last_claude_word_count.get(speaker, 0)

        # First analysis at threshold, then every interval after
        should_analyze = False
        if last_analyzed == 0 and word_count >= CLAUDE_ANALYSIS_THRESHOLD:
            should_analyze = True
        elif last_analyzed > 0 and (word_count - last_analyzed) >= CLAUDE_REANALYSIS_INTERVAL:
            should_analyze = True

        if should_analyze:
            # Combine all text from this speaker
            all_text = " ".join(profile.get("texts", []))
            if all_text:
                print(f"[Claude] Triggering analysis for {speaker} at {word_count} words...")
                insights = self._analyze_with_claude(speaker, all_text)
                if insights:
                    self.claude_insights[speaker] = insights
                    self.last_claude_word_count[speaker] = word_count

    def _process_audio_chunk(self, audio_data: np.ndarray, sample_rate: int):
        """Process an audio chunk - identify speaker and analyze."""
        result = {
            "speaker": "Unknown",
            "confidence": 0.0,
            "vak": None,
            "need": None,
            "tip": "",
            "text": "",
        }

        # Try to identify speaker by voice
        if self.identifier and len(audio_data) > sample_rate:  # At least 1 second
            try:
                embedding = self.identifier.extract_embedding(audio_data, sample_rate)
                if embedding is not None:
                    speaker, confidence = self._match_speaker(embedding)
                    result["speaker"] = speaker
                    result["confidence"] = confidence
            except Exception as e:
                pass  # Silent fail for real-time

        # Transcribe with Whisper
        try:
            import whisper
            import tempfile
            from scipy.io import wavfile

            # Save to temp file for whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                # Normalize and convert to int16
                audio_norm = audio_data / (np.abs(audio_data).max() + 1e-7)
                audio_int16 = (audio_norm * 32767).astype(np.int16)
                wavfile.write(temp_path, sample_rate, audio_int16)

            # Load whisper model (cached after first load)
            if not hasattr(self, '_whisper_model'):
                self._whisper_model = whisper.load_model("base")

            # Transcribe
            transcription = self._whisper_model.transcribe(temp_path, language="en")
            text = transcription.get("text", "").strip()
            result["text"] = text

            # Clean up
            Path(temp_path).unlink(missing_ok=True)

            # Analyze if we got text
            if text and len(text.split()) > 3:
                analysis = self.analyzer.analyze(text)

                # Get dominant VAK
                vak = analysis.get_dominant_vak()
                result["vak"] = vak

                # Get top social need
                top_needs = analysis.get_top_needs(1)
                if top_needs:
                    result["need"] = top_needs[0]

                # Update speaker profile
                speaker = result["speaker"]
                if speaker not in self.speaker_profiles:
                    self.speaker_profiles[speaker] = {
                        "vak_scores": {"visual": 0, "auditory": 0, "kinesthetic": 0},
                        "need_scores": {},
                        "word_count": 0,
                        "texts": [],
                    }

                # Accumulate scores
                profile = self.speaker_profiles[speaker]
                for mod, score in analysis.vak_scores.items():
                    profile["vak_scores"][mod] += score
                for need, score in analysis.social_needs.items():
                    profile["need_scores"][need] = profile["need_scores"].get(need, 0) + score
                profile["word_count"] += analysis.word_count
                profile["texts"].append(text)

                # Recalculate dominant VAK from accumulated scores
                total_vak = sum(profile["vak_scores"].values())
                if total_vak > 0:
                    dominant_vak = max(profile["vak_scores"], key=profile["vak_scores"].get)
                    result["vak"] = dominant_vak
                    profile["vak"] = dominant_vak

                # Recalculate dominant need
                if profile["need_scores"]:
                    dominant_need = max(profile["need_scores"], key=profile["need_scores"].get)
                    result["need"] = dominant_need
                    profile["need"] = dominant_need

                # Track deception/politician scores (smoothed average)
                pol_score = getattr(analysis, 'politician_score', 0)
                dec_score = getattr(analysis, 'deception_score', 0)
                dec_markers = getattr(analysis, 'deception_markers', {})

                if pol_score > 0 or dec_score > 0:
                    print(f"[Overlay] Deception: {dec_score:.2f} | Politician: {pol_score:.2f}")
                    if dec_markers:
                        print(f"[Overlay] Markers: {list(dec_markers.keys())}")

                if pol_score > 0:
                    old_pol = profile.get("politician_score", 0)
                    profile["politician_score"] = old_pol * 0.7 + pol_score * 0.3
                    result["politician_score"] = profile["politician_score"]
                if dec_score > 0:
                    old_dec = profile.get("deception_score", 0)
                    profile["deception_score"] = old_dec * 0.7 + dec_score * 0.3
                    result["deception_score"] = profile["deception_score"]
                if dec_markers:
                    result["deception_markers"] = dec_markers

                # Generate tip
                tips = []
                if result["vak"] in VAK_TIPS:
                    tips.append(VAK_TIPS[result["vak"]][0])
                if result["need"] in NEED_TIPS:
                    tips.append(NEED_TIPS[result["need"]][0])
                if tips:
                    result["tip"] = tips[0]

                # Trigger Claude analysis if threshold reached
                self._maybe_trigger_claude_analysis(speaker)

                # Add Claude insights to result if available
                if speaker in self.claude_insights:
                    result["claude_insights"] = self.claude_insights[speaker]

        except Exception as e:
            # Whisper not available or failed
            print(f"[Overlay] Transcription error: {e}")
            import traceback
            traceback.print_exc()

        self.current_speaker = result["speaker"]
        return result

    def _audio_callback(self, chunk):
        """Handle incoming audio."""
        self.audio_queue.put(chunk.data.copy())

    def _processing_loop(self):
        """Background processing loop."""
        buffer = []
        last_process_time = time.time()

        while self.running:
            try:
                # Collect audio chunks
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get_nowait()
                    buffer.append(chunk)

                # Process every 5 seconds (more audio = better transcription)
                if time.time() - last_process_time > 5 and buffer:
                    audio_data = np.concatenate(buffer)
                    # Use capture's sample rate
                    sample_rate = getattr(self.capture, 'sample_rate', 48000)
                    print(f"[Overlay] Processing {len(audio_data)/sample_rate:.1f}s of audio...")
                    result = self._process_audio_chunk(audio_data, sample_rate)
                    if result.get("text"):
                        print(f"[Overlay] Transcribed: {result['text'][:50]}...")
                    if result.get("vak"):
                        speaker = result.get("speaker", "Unknown")
                        if speaker in self.speaker_profiles:
                            profile = self.speaker_profiles[speaker]
                            vak_scores = profile.get("vak_scores", {})
                            need_scores = profile.get("need_scores", {})
                            # Show top 3 needs
                            top_needs = sorted(need_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"[Overlay] VAK scores: V={vak_scores.get('visual',0):.1f} A={vak_scores.get('auditory',0):.1f} K={vak_scores.get('kinesthetic',0):.1f}")
                            print(f"[Overlay] Top needs: {', '.join(f'{n}={s:.1f}' for n,s in top_needs)}")
                        print(f"[Overlay] => Dominant: VAK={result['vak']}, Need={result.get('need')}")
                    self.result_queue.put(result)
                    buffer = []  # Clear buffer after processing
                    last_process_time = time.time()

                time.sleep(0.1)
            except Exception as e:
                print(f"[Overlay] Processing error: {e}")
                import traceback
                traceback.print_exc()

    def start(self):
        """Start real-time profiling."""
        self.running = True

        # Start audio capture
        self.capture = AudioCapture(
            device_name=self.device,
            callback=self._audio_callback
        )
        self.capture.start_recording()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()

    def stop(self):
        """Stop profiling and save to database."""
        self.running = False
        if hasattr(self, 'capture'):
            self.capture.stop_recording()

        # Save profiles to database
        self._save_to_database()

    def _save_to_database(self):
        """Save session and speaker profiles to database."""
        if not self.speaker_profiles:
            print("[Overlay] No profiles to save.")
            return

        try:
            from .storage.database import init_db
            from .storage.models import Speaker, Session, Utterance, SpeakerProfile
            from datetime import datetime
            import uuid

            db = init_db()
            db_session = db.get_new_session()

            # Create session record
            session = Session(
                session_id=str(uuid.uuid4()),
                name=f"Overlay {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0,
            )
            db_session.add(session)
            db_session.commit()

            # Save each speaker's profile
            for speaker_name, profile in self.speaker_profiles.items():
                # Create or find speaker
                speaker = db_session.query(Speaker).filter_by(display_name=speaker_name).first()
                if not speaker:
                    speaker = Speaker(
                        speaker_id=f"overlay_{speaker_name.lower().replace(' ', '_')}_{session.id}",
                        display_name=speaker_name,
                    )
                    db_session.add(speaker)
                    db_session.commit()

                # Save utterance with all their text
                full_text = " ".join(profile.get("texts", []))
                if full_text:
                    # Include Claude insights in notes if available
                    notes = None
                    if speaker_name in self.claude_insights:
                        notes = json.dumps(self.claude_insights[speaker_name])

                    utterance = Utterance(
                        session_id=session.id,
                        speaker_db_id=speaker.id,
                        text=full_text,
                        start_time=0,
                        end_time=0,
                        word_count=profile.get("word_count", 0),
                        vak_detected=profile.get("vak"),
                        dominant_need=profile.get("need"),
                    )
                    # Store Claude insights if utterance has a notes field
                    if notes and hasattr(utterance, 'notes'):
                        utterance.notes = notes
                    db_session.add(utterance)

            db_session.commit()
            session_id = session.id  # Get ID before closing
            db_session.close()
            print(f"[Overlay] Saved {len(self.speaker_profiles)} speaker profiles to database (Session ID: {session_id})")

        except Exception as e:
            print(f"[Overlay] Failed to save to database: {e}")
            import traceback
            traceback.print_exc()

    def get_latest_result(self) -> Optional[dict]:
        """Get the latest profiling result."""
        result = None
        while not self.result_queue.empty():
            result = self.result_queue.get_nowait()
        return result


class OverlayWindow:
    """Transparent overlay window showing real-time profile."""

    def __init__(self, profiler: RealtimeProfiler):
        if not TK_AVAILABLE:
            raise ImportError("tkinter required for overlay")

        self.profiler = profiler
        self.root = tk.Tk()
        self._setup_window()
        self._create_widgets()

    def _setup_window(self):
        """Configure the overlay window."""
        self.root.title("Profiler")
        self.root.attributes('-topmost', True)  # Always on top
        self.root.attributes('-alpha', 0.85)  # Semi-transparent
        self.root.overrideredirect(True)  # No window decorations

        # Position in bottom-right corner
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.window_width = 380
        x = screen_width - self.window_width - 20
        y = screen_height - 450  # Start position, will auto-adjust

        self.root.geometry(f"{self.window_width}x400+{x}+{y}")
        self._screen_width = screen_width
        self._screen_height = screen_height
        self.root.configure(bg='#1a1a2e')

        # Track if user has manually positioned the window
        self._user_positioned = False
        self._user_x = None
        self._user_y = None

        # Track if we're closing
        self._closing = False

        # Make draggable
        self.root.bind('<Button-1>', self._start_drag)
        self.root.bind('<B1-Motion>', self._drag)

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")
        # Remember user's position
        self._user_positioned = True
        self._user_x = x
        self._user_y = y

    def _create_widgets(self):
        """Create overlay widgets."""
        # Main frame
        frame = tk.Frame(self.root, bg='#1a1a2e', padx=10, pady=10)
        frame.pack(fill='both', expand=True)

        # Header with close button
        header = tk.Frame(frame, bg='#1a1a2e')
        header.pack(fill='x')

        self.speaker_label = tk.Label(
            header,
            text="Listening...",
            font=('Segoe UI', 14, 'bold'),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        self.speaker_label.pack(side='left')

        close_btn = tk.Button(
            header,
            text="X",
            font=('Segoe UI', 10),
            fg='white',
            bg='#ff4757',
            bd=0,
            command=self.close
        )
        close_btn.pack(side='right')

        # Confidence
        self.confidence_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 9),
            fg='#888',
            bg='#1a1a2e'
        )
        self.confidence_label.pack(anchor='w')

        # Separator
        sep = tk.Frame(frame, height=1, bg='#333')
        sep.pack(fill='x', pady=8)

        # Profile info
        self.vak_label = tk.Label(
            frame,
            text="VAK: -",
            font=('Segoe UI', 11),
            fg='white',
            bg='#1a1a2e'
        )
        self.vak_label.pack(anchor='w')

        self.need_label = tk.Label(
            frame,
            text="Need: -",
            font=('Segoe UI', 11),
            fg='white',
            bg='#1a1a2e'
        )
        self.need_label.pack(anchor='w')

        # Politician indicator (hidden by default)
        self.politician_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 11, 'bold'),
            fg='#ff6b6b',
            bg='#1a1a2e'
        )
        self.politician_label.pack(anchor='w', pady=(5, 0))

        # Separator
        sep2 = tk.Frame(frame, height=1, bg='#333')
        sep2.pack(fill='x', pady=8)

        # Tip
        self.tip_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 10),
            fg='#ffd700',
            bg='#1a1a2e',
            wraplength=350,
            justify='left'
        )
        self.tip_label.pack(anchor='w')

        # Separator before Claude section
        self.claude_sep = tk.Frame(frame, height=1, bg='#444')
        self.claude_sep.pack(fill='x', pady=8)
        self.claude_sep.pack_forget()  # Hidden initially

        # Claude insight header
        self.claude_header = tk.Label(
            frame,
            text="🤖 Claude Analysis",
            font=('Segoe UI', 10, 'bold'),
            fg='#00ff88',
            bg='#1a1a2e'
        )
        self.claude_header.pack(anchor='w')
        self.claude_header.pack_forget()  # Hidden initially

        # Claude rapport tip
        self.claude_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 10),
            fg='#00ff88',
            bg='#1a1a2e',
            wraplength=350,
            justify='left'
        )
        self.claude_label.pack(anchor='w', pady=(3, 0))

        # Claude personality summary
        self.personality_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 9),
            fg='#cccccc',
            bg='#1a1a2e',
            wraplength=350,
            justify='left'
        )
        self.personality_label.pack(anchor='w', pady=(5, 0))

        # Claude persuasion tip
        self.persuade_label = tk.Label(
            frame,
            text="",
            font=('Segoe UI', 9),
            fg='#ff9f43',
            bg='#1a1a2e',
            wraplength=350,
            justify='left'
        )
        self.persuade_label.pack(anchor='w', pady=(5, 0))

    def _update(self):
        """Update the overlay with latest results."""
        result = self.profiler.get_latest_result()

        if result:
            # Update speaker
            speaker = result.get("speaker", "Unknown")
            confidence = result.get("confidence", 0)

            self.speaker_label.config(text=f"{speaker}")

            if confidence > 0 and confidence < 1.0:
                self.confidence_label.config(text=f"Voice match: {confidence:.0%}")
            else:
                self.confidence_label.config(text="")

            # Get VAK and Need from result or stored profile
            vak = result.get("vak")
            need = result.get("need")
            tip = result.get("tip", "")

            # Also check stored profile for accumulated data
            if speaker in self.profiler.speaker_profiles:
                profile = self.profiler.speaker_profiles[speaker]
                if not vak:
                    vak = profile.get("vak")
                if not need:
                    need = profile.get("need")
                word_count = profile.get("word_count", 0)
                self.confidence_label.config(
                    text=f"Words: {word_count}" + (f" | Match: {confidence:.0%}" if confidence > 0 and confidence < 1.0 else "")
                )

            # Update display
            if vak:
                vak_emoji = {"visual": "👁", "auditory": "👂", "kinesthetic": "✋"}.get(vak, "")
                self.vak_label.config(text=f"VAK: {vak.capitalize()} {vak_emoji}")
            else:
                self.vak_label.config(text="VAK: listening...")

            if need:
                self.need_label.config(text=f"Need: {need.capitalize()}")
            else:
                self.need_label.config(text="Need: listening...")

            # Show politician indicator
            politician_score = result.get("politician_score", 0)
            if speaker in self.profiler.speaker_profiles:
                politician_score = self.profiler.speaker_profiles[speaker].get("politician_score", politician_score)

            if politician_score > 0.5:
                self.politician_label.config(text=f"🏛️ Politician? {politician_score:.0%}", fg='#ff6b6b')
            elif politician_score > 0.3:
                self.politician_label.config(text=f"🤔 Evasive? {politician_score:.0%}", fg='#ffa502')
            else:
                self.politician_label.config(text="")

            # Show tip
            if tip:
                self.tip_label.config(text=f"💡 {tip}")
            elif vak or need:
                tips = []
                if vak and vak in VAK_TIPS:
                    tips.append(VAK_TIPS[vak][0])
                if need and need in NEED_TIPS:
                    tips.append(NEED_TIPS[need][0])
                if tips:
                    self.tip_label.config(text=f"💡 {tips[0]}")
            else:
                self.tip_label.config(text="")

            # Show Claude insights if available
            claude_insights = result.get("claude_insights")
            if not claude_insights and speaker in self.profiler.claude_insights:
                claude_insights = self.profiler.claude_insights[speaker]

            if claude_insights:
                # Show the Claude section
                self.claude_sep.pack(fill='x', pady=8)
                self.claude_header.pack(anchor='w')

                # Show rapport tip (full text)
                rapport_tip = claude_insights.get("rapport_tip", "")
                if rapport_tip:
                    self.claude_label.config(text=f"💬 Rapport: {rapport_tip}")
                else:
                    self.claude_label.config(text="")

                # Show personality summary (full text)
                personality = claude_insights.get("personality_summary", "")
                if personality:
                    self.personality_label.config(text=f"📋 Personality: {personality}")
                else:
                    self.personality_label.config(text="")

                # Show persuasion tip (full text)
                persuade = claude_insights.get("how_to_persuade", "")
                if persuade:
                    self.persuade_label.config(text=f"🎯 Persuade: {persuade}")
                else:
                    self.persuade_label.config(text="")

                # Auto-resize window to fit content
                self.root.update_idletasks()
                req_height = self.root.winfo_reqheight()
                req_width = max(self.root.winfo_reqwidth(), self.window_width)

                # Keep user's position if they dragged it, otherwise use bottom-right
                if self._user_positioned and self._user_x is not None:
                    x = self._user_x
                    y = self._user_y
                else:
                    x = self._screen_width - req_width - 20
                    y = self._screen_height - req_height - 60

                self.root.geometry(f"{req_width}x{req_height}+{x}+{y}")
            else:
                self.claude_sep.pack_forget()
                self.claude_header.pack_forget()
                self.claude_label.config(text="")
                self.personality_label.config(text="")
                self.persuade_label.config(text="")

        # Schedule next update (unless closing)
        if not self._closing:
            self.root.after(500, self._update)

    def run(self):
        """Run the overlay."""
        self.profiler.start()
        self._update()
        self.root.mainloop()

    def close(self):
        """Close the overlay."""
        print("[Overlay] Closing...")

        # Stop the update loop
        self._closing = True

        # Stop the profiler first
        self.profiler.stop()

        # Give processing thread time to finish
        if hasattr(self.profiler, 'process_thread') and self.profiler.process_thread.is_alive():
            print("[Overlay] Waiting for processing thread...")
            self.profiler.process_thread.join(timeout=2.0)

        # Destroy the window
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        print("[Overlay] Closed.")


def run_overlay(device: int):
    """Start the real-time overlay."""
    if not TK_AVAILABLE:
        print("Error: tkinter not available")
        return

    if not PYANNOTE_AVAILABLE:
        print("Warning: pyannote not available - speaker matching disabled")

    profiler = RealtimeProfiler(device)
    overlay = OverlayWindow(profiler)

    print("Overlay started. Drag to move, X to close.")
    overlay.run()
