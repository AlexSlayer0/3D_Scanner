import time
import depthai as dai

from argparse import ArgumentParser

NEURAL_FPS = 8
STEREO_DEFAULT_FPS = 30
TOF_DEFAULT_FPS = 30

parser = ArgumentParser()
parser.add_argument("--webSocketPort", type=int, default=8765)
parser.add_argument("--httpPort", type=int, default=8082)
parser.add_argument("--depthSource", type=str, default="stereo", choices=["stereo", "neural", "tof"])
args = parser.parse_args()

with dai.Pipeline() as p:
    remoteConnector = dai.RemoteConnection(
        webSocketPort=args.webSocketPort, httpPort=args.httpPort
    )

    size = (640, 400)
    if args.depthSource == "neural":
        fps = NEURAL_FPS
    elif args.depthSource == "tof":
        fps = TOF_DEFAULT_FPS
    else:
        fps = STEREO_DEFAULT_FPS

    if args.depthSource == "stereo":
        # 1. Create and configure Camera nodes (as before)
        color = p.create(dai.node.Camera).build(sensorFps=fps)
        left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
        right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)

        # 2. Get the output streams from the cameras
        left_stream = left.requestOutput(size)
        right_stream = right.requestOutput(size)

        # 3. Create and build StereoDepth, passing the streams directly
        depthSource = p.create(dai.node.StereoDepth).build(
            left=left_stream,
            right=right_stream,
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT
        )
        # Configure after building
        depthSource.setRectifyEdgeFillColor(0)
        depthSource.enableDistortionCorrection(True)

        # 4. Create and build the RGBD node (standalone)
        rgbd = p.create(dai.node.RGBD).build()
        # Optional: Configure RGBD if needed

        # 5. LINK the nodes
        color_output = color.requestOutput((640, 400), type=dai.ImgFrame.Type.RGB888i)
        color_output.link(rgbd.inColor)
        depthSource.depth.link(rgbd.inDepth) # Link stereo depth -> RGBD depth input
        
    elif args.depthSource == "neural":
        # 1. Create and configure Camera nodes
        color = p.create(dai.node.Camera).build(sensorFps=fps)
        left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=fps)
        right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)

        # 2. Get the output streams from the cameras
        left_stream = left.requestOutput(size)
        right_stream = right.requestOutput(size)

        # 3. Create and build NeuralDepth, passing the streams and model
        depthSource = p.create(dai.node.NeuralDepth).build(
            left_stream,
            right_stream,
            dai.DeviceModelZoo.NEURAL_DEPTH_LARGE
        )

        # 4. Create and build the RGBD node (standalone)
        rgbd = p.create(dai.node.RGBD).build()

        # 5. LINK the nodes: Get color stream and connect everything
        color_output = color.requestOutput(size, type=dai.ImgFrame.Type.RGB888i)
        color_output.link(rgbd.inColor)          # Color stream -> RGBD
        depthSource.depth.link(rgbd.inDepth)     # Neural depth -> RGBD
        
    elif args.depthSource == "tof":
        color = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=fps)
        socket, preset_mode = dai.CameraBoardSocket.AUTO, dai.ImageFiltersPresetMode.TOF_MID_RANGE
        depthSource = p.create(dai.node.ToF).build(socket, preset_mode)
    else:
        raise ValueError(f"Invalid depth source: {args.depthSource}")

    rgbd = p.create(dai.node.RGBD).build(color, depthSource, size, fps)

    remoteConnector.addTopic("pcl", rgbd.pcl, "common")
    p.start()
    remoteConnector.registerPipeline(p)

    print("Pipeline started with depth source: ", args.depthSource)

    while p.isRunning():
        key = remoteConnector.waitKey(1)
        if key == ord("q"):
            print("Got q key from the remote connection!")
            break
