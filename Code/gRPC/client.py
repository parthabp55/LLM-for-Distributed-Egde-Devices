#This script acts as a gRPC client on Jetson J1 (192.168.1.103) and fetches the current time from J3.

import grpc
import time_service_pb2
import time_service_pb2_grpc

def run():
    channel = grpc.insecure_channel("192.168.1.101:50051")  # Connect to J3
    stub = time_service_pb2_grpc.TimeServiceStub(channel)
    response = stub.GetCurrentTime(time_service_pb2.TimeRequest())
    print(f"🕒 Current Time from J3: {response.current_time}")

if __name__ == "__main__":
    run()


#Run this on J1 (Client) : python3 client.py

#Expected Output (on J1) : Current Time from J3: 2025-02-24 13:24:31


