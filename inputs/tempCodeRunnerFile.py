depth_input = cv.resize(frame, (depth_width, depth_height), interpolation=cv.INTER_AREA)

    rgb_frame = cv.cvtColor(depth_input, cv.COLOR_BGR2RGB)

    rgb_image = Image.fromarray(rgb_frame)
    start_time = time.perf_counter()
   
    result = depth_model(rgb_image)

    inference_time = time.perf_counter() - start_time
   
    depth = result["depth"]
   
    depth = np.array(depth)
   
    depth = cv.resize(depth, (frame.shape[1], frame.shape[0]))
