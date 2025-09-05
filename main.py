#!/usr/bin/env python
# coding: utf-8

# In[46]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("98176-647151506_small.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Could not read frame from video")


# In[47]:


plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("First Frame from Video")
plt.show()

cv2.imwrite("frame.png", frame)


# In[48]:


roi = cv2.selectROI("Select Traffic Light", frame, fromCenter=False, showCrosshair=True)

x, y, w, h = roi
crop = frame[y:y+h, x:x+w]
cv2.destroyAllWindows()

import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.title("Traffic Light ROI")
plt.show()


# In[49]:


hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV ROI")
plt.show()


# In[50]:


lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)
lower_red2 = (160, 100, 100)
upper_red2 = (180, 255, 255)

lower_yellow = (20, 100, 100)
upper_yellow = (30, 255, 255)

lower_green = (40, 100, 100)
upper_green = (90, 255, 255)

mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red  = cv2.bitwise_or(mask_red1, mask_red2)

mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_green  = cv2.inRange(hsv, lower_green, upper_green)


# In[51]:


plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(mask_red, cmap="gray"); plt.title("Red Mask")
plt.subplot(1,3,2); plt.imshow(mask_yellow, cmap="gray"); plt.title("Yellow Mask")
plt.subplot(1,3,3); plt.imshow(mask_green, cmap="gray"); plt.title("Green Mask")
plt.show()
mask_green = cv2.inRange(hsv, lower_green, upper_green)


# In[52]:


red_count    = cv2.countNonZero(mask_red)
yellow_count = cv2.countNonZero(mask_yellow)
green_count  = cv2.countNonZero(mask_green)
print("Red pixels:", red_count)
print("Yellow pixels:", yellow_count)
print("Green pixels:", green_count)


# In[53]:


if red_count > yellow_count and red_count > green_count:
    state = "🔴 Red Light is ON"
elif yellow_count > red_count and yellow_count > green_count:
    state = "🟡 Yellow Light is ON"
elif green_count > red_count and green_count > yellow_count:
    state = "🟢 Green Light is ON"
else:
    state = "⚠️ No clear light is detected."

print(state)


# In[54]:


cap = cv2.VideoCapture("98176-647151506_small.mp4")


# In[55]:


ret, frame = cap.read()
print(ret, frame.shape if ret else "No frame")


# In[56]:


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# In[57]:


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_detection.mp4", fourcc, 30, 
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_count = cv2.countNonZero(mask_red)
    yellow_count = cv2.countNonZero(mask_yellow)
    green_count = cv2.countNonZero(mask_green)

    if red_count > yellow_count and red_count > green_count:
        state, color = "🔴 Red Light  is ON", (0, 0, 255)
    elif yellow_count > red_count and yellow_count > green_count:
        state, color = "🟡 Yellow Light is ON", (0, 255, 255)
    elif green_count > red_count and green_count > yellow_count:
        state, color = "🟢 Green Light is ON", (0, 255, 0)
    else:
        state, color = "⚠️ No clear light is detected.", (255, 255, 255)

    cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 3)

    cv2.imshow("Traffic Light Detection", frame)
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:




