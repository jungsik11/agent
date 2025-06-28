import 'package:ollama_dart/ollama_dart.dart';

class AIService {
  static const String _baseUrl = 'http://localhost:11434/api';


  final OllamaClient _client = OllamaClient(baseUrl: _baseUrl);

  Future<List<Model>> getModels() async {
    final response = await _client.listModels();
    return response.models ?? [];
  }



  Stream<GenerateCompletionResponse> generateResponseStream(
      String model, String prompt, {int? numPredict}) {
    return _client.generateCompletionStream(
      request: GenerateCompletionRequest(
        model: model,
        prompt: prompt,
        options: numPredict != null ? RequestOptions(numPredict: numPredict) : null,
      ),
    );
  }
}
