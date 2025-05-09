**This file explains how to set up and run the project.**

**gRPC & REST API on Jetsons**

This project sets up **Jetson J3 (Server) & Jetson J1 (Client)** for time retrieval via **gRPC and REST API**.

**Setup Steps**

1. **Assign Static IPs**
- **J3 (Server):** `192.168.1.101`
- **J1 (Client):** `192.168.1.103`

Run this command on each Jetson:
- sudo nmcli con modify "Wired connection 1" ipv4.addresses 192.168.1.xxx/24 ipv4.gateway 192.168.1.1 ipv4.method manual
- sudo nmcli con up "Wired connection 1"

2. **Enable SSH**

- sudo systemctl enable ssh
- sudo systemctl start ssh


3. **Verify Connectivity**

- ping -c 4 192.168.1.101  #From J1 to J3


4. **Install Dependencies**

- pip install -r requirements.txt


5. **Start gRPC Server on J3**

- python3 server.py


6. **Run gRPC Client on J1**

- python3 client.py

**Expected Output:**
Current Time from J3: 2025-02-24 13:24:31



7. **Start REST API on J3**

- python3 rest_api.py

8. **Access Time via REST API**

- curl http://192.168.1.101:8000


**Expected Output:**
{"current_time":"2025-02-24 13:24:31"}

**Troubleshooting**

|              **Issue**              |                            **Fix**                          |
|     Destination Host Unreachable    |             Check network & static IP settings              |
|       Connection Refused (gRPC)	  |         Ensure server is running & port 50051 is open       |
|    Connection Refused (REST API)    |     	Ensure rest_api.py is running & port 8000 is open   |
|        No Module Named 'grpc'       |            Run : pip install grpcio grpcio-tools            |
|         Failed to Start SSH         |              Run : sudo systemctl restart ssh               |
	
grpc-jetson-setup/
│── server.py            # gRPC Server on J3
│── client.py            # gRPC Client on J1
│── time_service.proto   # gRPC Protocol Buffer
│── rest_api.py          # REST API for user interface
│── requirements.txt     # Dependencies
│── README.md            # Documentation
│── .gitignore           # Ignore unnecessary files

