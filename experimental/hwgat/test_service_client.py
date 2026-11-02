import sys
import os
import cv2
import io
import numpy as np

sys.path.insert(0, os.path.abspath('experimental/hwgat'))
import hwgat_service

client = hwgat_service.app.test_client()

print("=" * 60)
print("TESTING HWGAT REAL-TIME RECOGNITION SERVICE")
print("=" * 60)

# 1. Health Check
res_h = client.get('/health')
print("Health:", res_h.get_json())
assert res_h.status_code == 200

# 2. Sequence of 12 frames
print("\nSending stream of 12 video frames for User A...")
for i in range(12):
    # Simulated frame with hand drawing to induce non-zero landmarks
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (320 + i*5, 240 + i*2), 40, (50, 50, 200), -1)
    
    _, enc = cv2.imencode('.jpg', frame)
    data = {
        'frame': (io.BytesIO(enc.tobytes()), 'frame.jpg'),
        'user_id': 'user_a_tester',
        'threshold': '0.01'
    }
    res = client.post('/recognize_frame', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    j = res.get_json()
    print(f"Frame #{i+1:02d} -> text='{j.get('text')}', conf={j.get('confidence')}%, state={j.get('signing_state')}, msg='{j.get('message')}'")

# 3. Simultaneous Stream for User B
print("\nSending simultaneous stream for User B...")
data_b = {
    'frame': (io.BytesIO(enc.tobytes()), 'frame.jpg'),
    'user_id': 'user_b_tester',
    'threshold': '0.01'
}
res_b = client.post('/recognize_frame', data=data_b, content_type='multipart/form-data')
print("User B response:", res_b.get_json())

# 4. Reset User A
res_r = client.post('/reset_user', data={'user_id': 'user_a_tester'})
print("\nReset User A:", res_r.get_json())

print("=" * 60)
print("ALL HWGAT SERVICE PIPELINE TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)

