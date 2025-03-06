def start():
    global motor
    global image
    global Width, Height
    global img
    cam_record = False
    debug_mode = True  # Activated debugging mode
    save_debug_image = True

    rospy.init_node('auto_drive')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback)

    image_sub = rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
    print("---------- Xycar C1 HD v1.0 ----------")
    time.sleep(3)

    # sq = rospy.Rate(30)
    stage = 0
    t_check = time.time()
    f_n = 0
    p_angle = 0
    flag = 0
    line_count = 0
    avoid_time = time.time() + 3.8
    before_start = time.time() + 1
    turn_right = time.time()
    stop_time = time.time() + 16 + 1
    speed_time = 0
    b_angle = 0
    b_error = 0
    ITerm = 0
    Opt = 0 #obstacle
    Cnt = 0
    speed = 0
    w_time = False
    stage_time = 0

    if cam_record:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        path = '/home/pi/xycar_ws/src/base/cam_record'
        out = cv2.VideoWriter(os.path.join(path, 'test.avi'), fourcc, 25.0, (Width, Height))

    while not rospy.is_shutdown():

        while not image.size == (Width * Height * 3):
            continue

        if time.time() < before_start:
            drive(0, 0)
            continue

        f_n += 1
        if (time.time() - t_check) > 1:
            # print("fps : ", f_n)
            t_check = time.time()
            f_n = 0
        if cam_record:
            out.write(image)
        draw_img = image.copy()

        try:
            lpos, rpos, white_stop, red_mask, green_mask, yellow_mask = process_image(draw_img, w_time,stage)
        except Exception as e:
            print(f"An error occurred: {e}")
            lpos, rpos, white_stop, red_mask, green_mask, yellow_mask = process_image(draw_img, w_time,stage)

        stage = update_stage_based_on_color(red_mask, green_mask, yellow_mask, stage)
        # debugging image save
        if debug_mode and save_debug_image:
            filename = os.path.join(DEBUG_SAVE_PATH, f'image_{debug_counter}.png')
            cv2.imwrite(filename, draw_img)

        if time.time() > stop_time:
            w_time = True
        if w_time:
            if (white_stop >= 400):
                line_count += 1
                stop_time = time.time() + 11.5
                w_time = False

        # stop
        if (line_count == 2):  # 라인 카운트 2개시 정지
            drive(0, 0)
            # cv2.waitKey(0)
            line_count = 0

        if stage == 1 and time.time() > stage_time:
            stage = 2
            max_time_end = time.time() + 0.7  # start(True)
            while True:
                drive(100, 15)
                if time.time() > max_time_end:
                    break
            max_time_end = time.time() + 0.95  # go back to the line
            while True:
                drive(-100, 8)
                if time.time() > max_time_end:
                    break
            speed_time = time.time() + 1.5

        diff = rpos - lpos

        if diff > 135 and diff < 142:
            print("straight")
        else:
            print("curve")

        # color
        hsv = cv2.cvtColor(draw_img, cv2.COLOR_BGR2GRAY)
        # traffic light

        if (lpos == 0):
            # print("lpos error")
            lpos = rpos - 130
        if (rpos > lpos + 145):
            # print("rpos error")
            rpos = lpos + 130

        center = (lpos + rpos) / 2

        error = (center - Width / 2)
        angle, ITerm = pid_angle(ITerm, error, b_angle, b_error, Cnt)

        if stage == 0:
            drive(angle, speed)

        #Initial stage = 0 , if green light -> stage = 1
        #green and blue -> driving
        if stage == 0:
            if red_mask is not None and cv2.countNonZero(red_mask) > 1000:
                print("red right stop")
                max_time_end = time.time() + 3
                break
            elif yellow_mask is not None and cv2.countNonZero(yellow_mask) > 1000:
                print("yellow right stop")
                max_time_end = time.time() + 3
                break
        if stage == 1:
            if green_mask is not None and cv2.countNonZero(green_mask) > 1000:
                print("Green light detected during stage 1")
                max_time_end = time.time() + 3
                #If current position is curve -> curve
                #Else -> straight
                while True:
                    if diff > 135 and diff < 142:
                        print("straight")
                    else:
                        print("curve")
                    if time.time() > max_time_end:
                        break

        if (stage == 1 or time.time() < speed_time):
            speed = 3.8
        elif (stage == 0):
            # print("before avoid car")
            speed = 22
        elif angle < -17.5 or angle > 17.5:
            speed = 23
        else:
            speed = 30

        if angle < -17 or angle > 17:
            speed = 22
            #print("angle:", angle, "speed:", speed, "car_curve")
        else:
            speed = 30
            #print("angle:", angle, "speed", speed, "car_straight")

        drive(angle, 22)

        # cv2.waitKey(1)
        # sq.sleep()
        b_angle = angle
        b_error = error

