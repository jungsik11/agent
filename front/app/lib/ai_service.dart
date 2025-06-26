import 'dart:developer';

import 'package:ollama_dart/ollama_dart.dart';

class AIService {
  static const String _baseUrl = 'http://localhost:11434/api';
  static const String _model = 'local-model1';

  final OllamaClient _client = OllamaClient(baseUrl: _baseUrl);



  // 번역 X
  Future<String> generateResponse(String prompt) async {
    final stopwatch = Stopwatch()..start();
  
     final aiResponse = await _generateAIResponse(prompt);
     final aiGenerationTime = stopwatch.elapsedMilliseconds;
  
     log('AI Generation Time: ${aiGenerationTime}ms', name: 'AIService');
  
     return aiResponse;
   }


  Future<String> _generateAIResponse(String prompt) async {
    final stream = _client.generateCompletionStream(
      request: GenerateCompletionRequest(
        model: _model,
        prompt: prompt,
      ),
    );

    String fullResponse = '';
    await for (final chunk in stream) {
      final response = chunk.response ?? '';
      for (final char in response.split('')) {
        fullResponse += char;
      }
    }

    return fullResponse;
  }
}
