import depthai as dai

pipeline = dai.Pipeline()
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("depth")
stereo.depth.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("depth", maxSize=4, blocking=True)
    frame = q.get().getCvFrame()
    print("Min/Max Depth:", frame.min(), frame.max())