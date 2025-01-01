#include <iostream>
#include <thread>
#include <chrono>
#include <steam/steamnetworkingsockets.h>

// Function to initialize the library and connect the client to the server
void RunClient(const char* serverAddressString)
{
    // Initialize GameNetworkingSockets
    SteamDatagramErrMsg errMsg;
    if (!GameNetworkingSockets_Init(nullptr, errMsg)) {
        std::cerr << "Failed to initialize GameNetworkingSockets: " << errMsg << '\n';
        return;
    }

    // Create the socket interface for the client
    ISteamNetworkingSockets* networkingSockets = SteamNetworkingSockets();
    SteamNetworkingIPAddr serverAddress;
    serverAddress.Clear();
    
    serverAddress.m_port = 27015;

    // Connect to the server
    HSteamNetConnection clientConnection = networkingSockets->ConnectByIPAddress(serverAddress, 0, nullptr);
    if (clientConnection == k_HSteamNetConnection_Invalid) {
        std::cerr << "Failed to connect to server" << '\n';
        return;
    }

    std::cout << "Connected to server at " << serverAddressString << '\n';

    // Send a message
    const char* message = "Hello from the client!";
    networkingSockets->SendMessageToConnection(clientConnection, message, strlen(message), k_nSteamNetworkingSend_Reliable, nullptr);

    // Event loop to process incoming messages
    while (true) {
        SteamNetworkingMessage_t* incomingMessages[10];
        int numMessages = networkingSockets->ReceiveMessagesOnConnection(clientConnection, incomingMessages, 10);

        for (int i = 0; i < numMessages; i++) {
            std::cout << "Received message: " << (char*)incomingMessages[i]->m_pData << '\n';
            incomingMessages[i]->Release();
        }

        // Sleep to avoid 100% CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup
    networkingSockets->CloseConnection(clientConnection, 0, nullptr, false);
    GameNetworkingSockets_Kill();
}

int main()
{
    const char* serverAddress = "127.0.0.1";
    std::cout << "Starting client..." << '\n';
    RunClient(serverAddress);
    return 0;
}
