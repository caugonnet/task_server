// server.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <cstring>
#include <netinet/in.h>
#include <unistd.h>
#include <cuda_runtime.h>

struct Task {
    Task() = default;

    Task(std::vector<int> _data, int client_socket) : data(::std::move(_data)), client_socket(client_socket) {
        result.resize(data.size());
    }

    std::vector<int> data;
    int client_socket;
    std::vector<int> result;
};

std::queue<Task> task_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;

// CUDA kernel
__global__ void double_elements(int* d_input, int* d_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < N) {
        d_output[idx] = 2 * d_input[idx];
        idx += blockDim.x*gridDim.x;
    }
}

// CUDA callback for notifying completion
void CUDART_CB completion_callback(cudaStream_t, cudaError_t, void* t)
{
    Task *task = reinterpret_cast<Task *>(t);

    // Send result back to client
    send(task->client_socket, task->result.data(), task->result.size() * sizeof(int), 0);

    const char* msg = "Done";
    send(task->client_socket, msg, strlen(msg), 0);
}

// Computation thread function
void computation_thread_func() {
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, []{ return !task_queue.empty(); });
            task = task_queue.front();
            task_queue.pop();
        }

        std::cout << "Processing data: ";
        for (int x : task.data)
            std::cout << x << " ";
        std::cout << "\n";

        int N = task.data.size();
        int* d_input;
        int* d_output;
        cudaMallocAsync(&d_input, N * sizeof(int), stream);
        cudaMallocAsync(&d_output, N * sizeof(int), stream);
        cudaMemcpyAsync(d_input, task.data.data(), N * sizeof(int), cudaMemcpyHostToDevice, stream);

        double_elements<<<64, 128, 0, stream>>>(d_input, d_output, N);

        cudaMemcpyAsync(task.result.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamAddCallback(stream, completion_callback, &task, 0);

        cudaFreeAsync(d_input, stream);
        cudaFreeAsync(d_output, stream);

        cudaStreamSynchronize(stream);
    }
}

// Client handler thread
void client_thread(int client_socket) {
    while (true) {
        int size;
        if (recv(client_socket, &size, sizeof(int), 0) <= 0) break;

        std::vector<int> data(size);
        if (recv(client_socket, data.data(), size * sizeof(int), 0) <= 0) break;

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.push({data, client_socket});
        }
        queue_cv.notify_one();
    }
    close(client_socket);
}

int main() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(12345);

    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    listen(server_fd, 10);
    std::cout << "Server listening on port 12345...\n";

    std::thread computation_thread(computation_thread_func);
    computation_thread.detach();

    while (true) {
        int client_socket = accept(server_fd, nullptr, nullptr);
        std::cout << "Client connected.\n";
        std::thread(client_thread, client_socket).detach();
    }

    close(server_fd);
    return 0;
}
