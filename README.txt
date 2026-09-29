ทดลองความเร็ว PaddleOCR-VL-1.5 บนโน้ตบุ๊ก Intel (OpenVINO)
============================================================

จุดประสงค์: วัดว่าเครื่อง Intel Core Ultra (GPU ในตัว / CPU) รันโมเดลนี้ได้เร็วแค่ไหน
เป็น "การทดลองแยก" — ไม่ใช่ผลที่ใช้เทียบคะแนน OmniDocBench (94.93) และห้ามเอาไปปนกับผลบน Colab
  - ใช้โหมด OCR ทั้งหน้าแบบ notebook ทางการของ OpenVINO (ไม่มีขั้นตรวจ layout)
  - ค่าเริ่มต้นบีบอัดโมเดลเป็น INT8 (ตาม notebook ทางการ)
  - ไม่ใช้ NPU (notebook ทางการตัด NPU ออก)

ก่อนเริ่ม: เครื่องนี้เป็นเครื่องบริษัท — ขออนุญาตพี่/IT ก่อนติดตั้ง
ต้องใช้: อินเทอร์เน็ต, พื้นที่ว่างประมาณ 10 GB, เวลาติดตั้งครั้งแรก 20-40 นาที

วิธีรัน
-------
1. copy โฟลเดอร์ openvino_test ทั้งโฟลเดอร์ไปไว้ในเครื่อง
   แนะนำ path สั้นๆ ไม่มีภาษาไทย เช่น C:\openvino_test
2. ดับเบิลคลิก setup.bat  (ครั้งแรกครั้งเดียว)
   - ติดตั้ง uv + Python 3.12 + PyTorch CPU + OpenVINO
   - โหลดโมเดล PaddleOCR-VL-1.5 (~2 GB) แล้วแปลงเป็น OpenVINO
   - จบแล้วขึ้น "[setup] DONE"
   - บรรทัด "devices:" ควรมี 'CPU', 'GPU' (และอาจมี 'NPU')
3. ดับเบิลคลิก run_test.bat
   - รัน 20 หน้าแรกของ OmniDocBench v1.6 บน GPU ในตัว แล้วรันซ้ำบน CPU
   - 20 หน้านี้เป็นหน้าเดียวกับชุดที่ 1 บน Colab เทียบความเร็วได้
4. ส่งกลับ: ข้อความ ===== SUMMARY ===== ทั้ง 2 ก้อน
   ผลแต่ละหน้าอยู่ใน results\gpu_int8\ และ results\cpu_int8\ (มี timing.csv)

ถ้าเจอ error
------------
- "An Application Control policy has blocked this file" = Windows ของเครื่องบริษัทบล็อกไฟล์
  → รันต่อไม่ได้ ต้องให้ IT อนุญาต (ห้ามไปปิดการตั้งค่าความปลอดภัยเอง)
- อย่างอื่น: copy ข้อความ error ส่งกลับมา

รันเพิ่มเติม (ไม่บังคับ)
------------------------
เปิด Command Prompt ในโฟลเดอร์นี้แล้วพิมพ์:
  .venv\Scripts\python.exe run_test.py --device GPU --n 20 --llm fp     (ไม่บีบอัด INT8)
  .venv\Scripts\python.exe run_test.py --device GPU --n 50              (50 หน้า = ชุดที่ 1 เต็ม)

ลบทิ้งทั้งหมด: ลบโฟลเดอร์ openvino_test (uv ที่ติดตั้งอยู่ใน %USERPROFILE%\.local\bin)
