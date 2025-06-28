import 'package:flutter/material.dart';

import 'package:app/message_bubble.dart';

class ChatMessage {
  final bool isUser;
  final String content;

  ChatMessage({required this.isUser, required this.content});
}

class ChatMessageList extends StatelessWidget {
  final List<ChatMessage> messages;
  final ScrollController scrollController;

  const ChatMessageList({
    super.key,
    required this.messages,
    required this.scrollController,
  });

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      controller: scrollController,
      itemCount: messages.length,
      itemBuilder: (context, index) {
        final isLatestMessage = index == messages.length - 1 && !messages[index].isUser;
        return MessageBubble(
          key: ValueKey('message_$index'),
          isUser: messages[index].isUser,
          message: messages[index].content,
          isLatestMessage: isLatestMessage,
        );
      },
    );
  }
}

class ChatInputField extends StatelessWidget {
  final TextEditingController controller;
  final bool isProcessing;
  final VoidCallback onSubmitted;

  const ChatInputField({
    super.key,
    required this.controller,
    required this.isProcessing,
    required this.onSubmitted,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          children: [
            Expanded(
              child: TextField(
                controller: controller,
                decoration: const InputDecoration(
                  hintText: 'Enter your message...',
                  border: OutlineInputBorder(),
                ),
                onSubmitted: (_) => onSubmitted(),
              ),
            ),
            const SizedBox(width: 8),
            isProcessing
                ? const CircularProgressIndicator()
                : IconButton(
              icon: const Icon(Icons.send),
              onPressed: onSubmitted,
            ),
          ],
        ),
      ),
    );
  }
}