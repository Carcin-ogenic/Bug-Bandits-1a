# Bug-Bandits-1A - PDF Document Structure Extraction

This solution automatically extracts hierarchical document structures from PDF files and generates JSON outlines using machine learning classification.

## Quick Start

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t bug-bandits-1a:latest .
```

### Run the Container
```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none bug-bandits-1a:latest
```

## Expected Behavior

The container will:
- Automatically process all PDFs from `/app/input` directory
- Generate corresponding `filename.json` files in `/app/output` for each `filename.pdf`
- Extract document structure with heading levels (H1, H2, H3, etc.) and page numbers
- Run in network isolation for security

## Output Format

Each generated JSON file contains:
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Heading Text",
      "page": 1
    }
  ]
}
```

## Technical Details

- **Base Image**: Python 3.9-slim
- **Platform**: linux/amd64
- **Dependencies**: PyMuPDF, ONNX Runtime, scikit-learn
- **Model**: Pre-trained text classification pipeline with multilingual embeddings