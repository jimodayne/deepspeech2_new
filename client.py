import socket

HOST = '34.80.20.135'    # Cấu hình address server
PORT = 8080              # Cấu hình Port sử dụng
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Cấu hình socket
s.connect((HOST, PORT)) # tiến hành kết nối đến server
s.sendall(b'Hello server!') # Gửi dữ liệu lên server 
data = s.recv(1024) # Đọc dữ liệu server trả về
s= data.decode("utf-8")
# s = unicode(data).encode('utf8')
print('Server Respone: ', s)