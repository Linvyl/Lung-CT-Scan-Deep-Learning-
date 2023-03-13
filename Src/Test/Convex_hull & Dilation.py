import cv2

# Đọc hình ảnh CT scan và chuyển đổi sang ảnh xám
img = cv2.imread('ct_scan.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Áp dụng phương pháp ngưỡng Otsu để trích xuất mặt nạ phổi
ret, lung_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Làm sạch mặt nạ phổi
lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20)))

# Tìm vùng bao lồi của mặt nạ phổi
lung_contours, _ = cv2.findContours(lung_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
lung_hull = cv2.convexHull(lung_contours[0])

# Dilate mặt nạ phổi
lung_mask = cv2.dilate(lung_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

# Hiển thị kết quả
cv2.drawContours(img, [lung_hull], 0, (0,255,0), 2)
cv2.imshow('Lung mask with convex hull', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
