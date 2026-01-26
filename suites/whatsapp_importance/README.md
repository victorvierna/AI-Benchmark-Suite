# whatsapp_importance

Benchmark for WhatsApp message importance classification.

## Run

```bash
benchkit run suites/whatsapp_importance/suite.yaml --models models.example.yaml --pricing pricing/openai.yaml
```

## Notes

- Expected output is strict JSON with fields: importance, isAutoReply, reason.
- Scoring is binary (pass/fail) based on importance + isAutoReply.
