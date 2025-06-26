import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:app/ai_service.dart';
import 'package:app/chat.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AIService _aiService = AIService();
  final TextEditingController _promptController = TextEditingController();
  final List<ChatMessage> _messages = [];
  final ScrollController _scrollController = ScrollController();
  bool _isProcessing = false;

  @override
  void dispose() {
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Ollama AI Chat")),
      body: Column(
        children: [
          Expanded(
            child: ChatMessageList(
              messages: _messages,
              scrollController: _scrollController,
            ),
          ),
          ChatInputField(
            controller: _promptController,
            isProcessing: _isProcessing,
            onSubmitted: _handleSubmission,
          ),
        ],
      ),
    );
  }

  Future<int> getMemoryUsage() async {
    try {
      const MethodChannel platform = MethodChannel('samples.flutter.dev/memory');
      final int result = await platform.invokeMethod('getMemoryUsage');
      return result;
    } on PlatformException catch (e) {
      print("Failed to get memory usage: '${e.message}'.");
      return 0;
    }
  }

  Future<void> _handleSubmission() async {
    if (_promptController.text.isEmpty) return;

    setState(() {
      _isProcessing = true;
      _messages.add(ChatMessage(isUser: true, content: _promptController.text));
    });

    try {
      final stopwatch = Stopwatch()..start();
      final response = await _aiService.generateResponse(_promptController.text);
      final totalProcessTime = stopwatch.elapsedMilliseconds;

      log('Total Process Time: ${totalProcessTime}ms', name: 'HomeScreen');

      setState(() {
        _messages.add(ChatMessage(isUser: false, content: response));
      });
    } catch (e) {
      setState(() {
        _messages.add(ChatMessage(isUser: false, content: '오류: $e'));
      });
      log('Error: $e', name: 'HomeScreen', error: e);
    } finally {
      setState(() {
        _isProcessing = false;
      });
      _promptController.clear();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
}