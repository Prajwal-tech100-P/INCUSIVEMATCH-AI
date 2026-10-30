import os
import re

def test_chat_interface_isl_video_call():
    chat_path = os.path.join("templates", "chat.html")
    assert os.path.exists(chat_path), "chat.html must exist"
    
    with open(chat_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Header has dedicated ISL Video Call button
    assert 'class="btn btn-primary btn-sm rounded-pill js-start-call' in content, "Header button should have primary styling and js-start-call class"
    assert 'title="Start ISL Video Call"' in content, "Header button title must indicate ISL Video Call"
    assert 'aria-label="Start Indian Sign Language Video Call' in content, "Header button must have accessible ISL label"
    assert 'ISL Video Call' in content, "Header button text must say 'ISL Video Call'"
    assert '<i class="bi bi-camera-video-fill"></i>' in content, "Header button icon must be bi-camera-video-fill"

    # 2. Bottom camera button removed completely
    assert 'id="btn-camera"' not in content, "Bottom camera button #btn-camera must be removed completely"
    assert 'Open Sign Language Camera' not in content, "Old camera tooltip/aria-label must be removed"

    # 3. Old video container removed
    assert 'id="video-container"' not in content, "Old local video container #video-container must be removed"
    assert 'id="webcam-feed"' not in content, "#webcam-feed must be removed"
    assert 'id="sign-text-overlay"' not in content, "#sign-text-overlay must be removed"

    # 4. Accessibility buttons preserved
    assert 'id="btn-speech"' in content, "#btn-speech (Mic: Speech-to-Text) must be preserved"
    assert 'id="btn-tts"' in content, "#btn-tts (Speaker: Text-to-Speech) must be preserved"
    assert 'id="btn-send"' in content, "#btn-send (Send Message) must be preserved"

    # 5. Placeholder updated cleanly
    assert 'placeholder="Type a message..."' in content, "Input placeholder must be 'Type a message...'"
    assert 'placeholder="Type a message or sign with camera..."' not in content, "Old placeholder must be removed"

    # 6. Dead camera script removed
    assert 'captureAndRecognizeFrame' not in content, "Old chat-embedded recognition loop must be removed"
    assert 'startCamera' not in content, "Old startCamera JS must be removed"

    print("ALL CHAT INTERFACE ASSERTIONS PASSED!")

def test_main_js_isl_call_labels():
    main_path = os.path.join("static", "js", "main.js")
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "ISL Video Call" in content, "main.js must display 'ISL Video Call'"
    assert "Incoming ISL Video Call" in content, "main.js must denote 'Incoming ISL Video Call'"
    print("ALL MAIN.JS ISL CALL LABELS PASSED!")

def test_compare_v1_vs_v2_import():
    compare_path = os.path.join("scratch", "compare_v1_vs_v2.py")
    with open(compare_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert "from scratch.run_nlp_audit import EVAL_DATASET" in content, "Must import from scratch.run_nlp_audit"
    print("ALL IMPORT TESTS PASSED!")

if __name__ == "__main__":
    test_chat_interface_isl_video_call()
    test_main_js_isl_call_labels()
    test_compare_v1_vs_v2_import()
    print("\n--- ALL VERIFICATION SUITES COMPLETED SUCCESSFULLY ---")

