# whatsapp_importance_classifier

Benchmark de clasificacion de mensajes de WhatsApp para un negocio de reservas.

## Objetivo

El modelo debe devolver JSON estricto con:

- `importance`: `important` o `not_important`
- `isAutoReply`: `true` si el mensaje parece una respuesta automatica
- `reason`: explicacion breve

La puntuacion compara exactamente `importance` e `isAutoReply`.

## Ejecutar

```bash
benchkit validate suites/whatsapp_importance_classifier/suite.yaml --models models.classifier.yaml --pricing pricing/cloud.yaml
benchkit run suites/whatsapp_importance_classifier/suite.yaml --models models.classifier.yaml --pricing pricing/cloud.yaml --runs 1 --report html
```

Para ejecutar todos los modelos de referencia:

```bash
benchkit run suites/whatsapp_importance_classifier/suite.yaml --models models.cloud.yaml --pricing pricing/cloud.yaml --runs 1 --report html
```

## Variables de entorno

- `OPENAI_API_KEY` para modelos OpenAI
- `GEMINI_API_KEY` para modelos Gemini
- `ANTHROPIC_API_KEY` para modelos Anthropic
