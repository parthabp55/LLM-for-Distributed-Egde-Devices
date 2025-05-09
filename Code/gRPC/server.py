#This script runs a gRPC server on Jetson J3 (192.168.1.101) and provides the current time to clients.

import grpc
import time
from concurrent import futures
import time_service_pb2
import time_service_pb2_grpc

class TimeServiceServicer(time_service_pb2_grpc.TimeServiceServicer):
    def GetCurrentTime(self, request, context):
        return time_service_pb2.TimeResponse(current_time=time.strftime("%Y-%m-%d %H:%M:%S"))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    time_service_pb2_grpc.add_TimeServiceServicer_to_server(TimeServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC Server running on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()


# Run this on J3 (Server): python3 server.py
  