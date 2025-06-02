// client.cpp
#include <iostream>
#include <vector>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int input_size = 5;
    if (argc > 1) {
        input_size = std::atoi(argv[1]);
    }

    std::vector<int> data(input_size);
    for (int i = 0; i < input_size; ++i)
        data[i] = i + 1;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("Connect failed");
        return 1;
    }

    int size = data.size();

    std::cout << "Data: ";
    for (int x : data)
        std::cout << x << " ";
    std::cout << "\n";


    send(sock, &size, sizeof(int), 0);
    send(sock, data.data(), size * sizeof(int), 0);

    std::vector<int> result(size);
    recv(sock, result.data(), size * sizeof(int), MSG_WAITALL);
    char buffer[5];
    recv(sock, buffer, 4, 0); // Receive "Done" message

    fprintf(stderr, "ACK? %d%d%d%d\n", buffer[0], buffer[1], buffer[2], buffer[3]);

    std::cout << "Result: ";
    for (int x : result)
        std::cout << x << " ";
    std::cout << "\n";

    close(sock);
    return 0;
}

