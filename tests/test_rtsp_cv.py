# file: test_rtsp_opencv.py
import cv2, time

URL = "rtsp://127.0.0.1:8554/test"   # 你的地址
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

# 可选：给 FFMPEG 传参（降低缓存、用 TCP）
# 有的构建不支持 set() 这些属性，失败也没关系
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)   # 小缓冲
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"H264"))  # 需要时再打开

if not cap.isOpened():
    raise SystemExit("❌ 打不开流，请检查地址/端口/服务器是否在监听")

win = "rtsp_view"
cv2.namedWindow(win)
t0, n = time.time(), 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️ 读取失败，尝试继续…")
        # 短暂等待，给解码器缓一缓
        if cv2.waitKey(10) == 27:
            break
        continue

    n += 1
    if n % 30 == 0:
        dt = time.time() - t0
        fps = n / dt if dt > 0 else 0
        print(f"✅ recv frames: {n}, approx FPS: {fps:.1f}")

    cv2.imshow(win, frame)
    # 按 q 或 Esc 退出
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
