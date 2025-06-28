import 'dart:async';
import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:app/ai_service.dart';
import 'package:app/chat.dart';
import 'package:ollama_dart/ollama_dart.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final AIService _aiService = AIService();
  final TextEditingController _promptController = TextEditingController();
  final TextEditingController _maxTokensController = TextEditingController();
  final List<ChatMessage> _messages = [];
  final ScrollController _scrollController = ScrollController();
  bool _isProcessing = false;

  List<Model> _models = [];
  String? _selectedModel;

  StreamSubscription<GenerateCompletionResponse>? _responseSubscription;

  @override
  void initState() {
    super.initState();
    _maxTokensController.text = '512';
    _fetchModels();
  }

  Future<void> _fetchModels() async {
    try {
      final models = await _aiService.getModels();
      if (mounted && models.isNotEmpty) {
        setState(() {
          _models = models;
          _selectedModel = models.first.model;
        });
      }
    } catch (e) {
      log('Failed to fetch models: $e', name: 'HomeScreen');
    }
  }

  @override
  void dispose() {
    _responseSubscription?.cancel();
    _promptController.dispose();
    _maxTokensController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Ollama AI Chat"),
        actions: [
          SizedBox(
            width: 100,
            child: TextField(
              controller: _maxTokensController,
              keyboardType: TextInputType.number,
              inputFormatters: [FilteringTextInputFormatter.digitsOnly],
              decoration: const InputDecoration(
                labelText: 'Max Tokens',
                labelStyle: TextStyle(color: Colors.black),
                enabledBorder: UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.white70),
                ),
                focusedBorder: UnderlineInputBorder(
                  borderSide: BorderSide(color: Colors.white),
                ),
              ),
              style: const TextStyle(color: Colors.black, fontSize: 14),
              textAlign: TextAlign.center,
            ),
          ),
          const SizedBox(width: 16),
          if (_models.isNotEmpty)
            DropdownButton<String>(
              value: _selectedModel,
              dropdownColor: Colors.white,
              style: const TextStyle(color: Colors.black),
              iconEnabledColor: Colors.black,
              underline: Container(),
              onChanged: (String? newValue) {
                setState(() {
                  _selectedModel = newValue;
                });
              },
              items: _models
                  .where((model) => model.model != null)
                  .map<DropdownMenuItem<String>>((Model model) {
                return DropdownMenuItem<String>(
                  value: model.model!,
                  child: Text(model.model!),
                );
              }).toList(),
            ),
          const SizedBox(width: 8),
        ],
      ),
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
      const MethodChannel platform =
          MethodChannel('samples.flutter.dev/memory');
      final int result = await platform.invokeMethod('getMemoryUsage');
      return result;
    } on PlatformException catch (e) {
      print("Failed to get memory usage: '${e.message}'.");
      return 0;
    }
  }

  Future<void> _handleSubmission() async {
    if (_promptController.text.isEmpty || _selectedModel == null) return;

    await _responseSubscription?.cancel();

    final prompt = _promptController.text;
    _promptController.clear();

    final maxTokens = int.tryParse(_maxTokensController.text);

    setState(() {
      _isProcessing = true;
      _messages.add(ChatMessage(isUser: true, content: prompt));
      _messages.add(ChatMessage(isUser: false, content: ''));
    });
    _scrollToBottom();

    _responseSubscription = _aiService
        .generateResponseStream(_selectedModel!, prompt, numPredict: maxTokens)
        .listen(
      (chunk) {
        setState(() {
          final lastMessage = _messages.last;
          final updatedContent = lastMessage.content + (chunk.response ?? '');
          _messages[_messages.length - 1] =
              ChatMessage(isUser: false, content: updatedContent);
        });
        _scrollToBottom();
      },
      onDone: () {
        setState(() {
          _isProcessing = false;
        });
      },
      onError: (e) {
        setState(() {
          _messages[_messages.length - 1] =
              ChatMessage(isUser: false, content: '오류: $e');
          _isProcessing = false;
        });
        log('Error: $e', name: 'HomeScreen', error: e);
      },
      cancelOnError: true,
    );
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