# CV Extract RAG - Educational CV Parser

A comprehensive web application that demonstrates Retrieval-Augmented Generation (RAG) concepts through an interactive CV parsing tool. This project serves as both a practical CV extraction system and an educational showcase for understanding how RAG works under the hood.

## 🎯 Project Goals

- **Educational Showcase**: Provide an interactive, step-by-step explanation of RAG concepts (document chunking, embedding, retrieval, LLM response)
- **Portfolio Project**: Highlight ability to design and implement RAG systems with clear explanations
- **Practical Application**: Offer a functional CV parsing tool with structured data extraction

## ✨ Features

### Core CV Extraction
- **Structured Data Extraction**: Parse CVs into organized sections (education, employment, skills, languages, certifications)
- **Multiple LLM Support**: OpenAI, Ollama, Scaleway, and Outlines adapters
- **Flexible Input**: Support for text-based CV input

### RAG Pipeline Visualization
- **Document Chunking**: Interactive view of how CVs are split into manageable chunks
- **Embedding Generation**: Visualization of vector embeddings with statistics
- **Semantic Retrieval**: See which chunks are most relevant to specific queries
- **LLM Response Generation**: Watch how retrieved context is used to generate answers

### Educational Features
- **Step-by-Step Process**: Visual indicators showing progress through the RAG pipeline
- **Technical Explanations**: Detailed explanations of each RAG component
- **Interactive Demo**: Hands-on experience with RAG concepts
- **Fallback Support**: TF-IDF embeddings when OpenAI API is unavailable

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for enhanced embeddings)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd CV-Extractor-RAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

## 📖 Usage

### Basic CV Extraction
1. Go to the main page
2. Paste your CV text into the textarea
3. Click "Extract CV Data"
4. View structured results in organized tables

### Interactive RAG Demo
1. Click "🚀 Try Interactive RAG Demo"
2. Upload or paste CV text
3. Watch the step-by-step RAG process:
   - Document chunking with overlap
   - Embedding generation
   - Query processing and retrieval
   - Final answer generation
4. Ask questions about the CV and see how RAG retrieves relevant information

## 🏗️ Architecture

### Backend Components
- **Flask Web Server**: Main application framework
- **RAG Service**: Core RAG pipeline implementation
- **LLM Adapters**: Multiple language model integrations
- **Data Processing**: Pandas for structured data handling

### RAG Pipeline
1. **Document Preprocessing**: Text cleaning and preparation
2. **Chunking**: Split documents into overlapping segments
3. **Embedding Generation**: Convert chunks to vector representations
4. **Query Processing**: Generate embeddings for user questions
5. **Semantic Retrieval**: Find most relevant chunks using similarity
6. **Context Assembly**: Prepare retrieved information for LLM
7. **Answer Generation**: Generate comprehensive responses using context

### Frontend Features
- **Responsive Design**: Modern, mobile-friendly interface
- **Interactive Visualizations**: Step-by-step process indicators
- **Educational Content**: Explanations for each RAG component
- **Real-time Updates**: Dynamic content loading and display

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for enhanced embeddings
- `FLASK_SECRET_KEY`: Secret key for Flask sessions (change in production)

### Customization
- **Chunk Size**: Adjust document chunking parameters in `rag_service.py`
- **Embedding Models**: Modify embedding generation in the RAG service
- **LLM Providers**: Switch between different language model adapters

## 📚 Educational Value

This project demonstrates key RAG concepts:

- **Document Chunking**: Why and how to split documents for processing
- **Vector Embeddings**: Converting text to numerical representations
- **Semantic Search**: Finding relevant information using similarity
- **Context Assembly**: Preparing retrieved information for LLMs
- **Response Generation**: Combining retrieval with language model reasoning

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Pandas, NumPy, Scikit-learn
- **AI/ML**: OpenAI API, TF-IDF fallback, Cosine similarity
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Data Processing**: Pandas DataFrames, JSON serialization
- **Deployment**: Flask development server (production-ready with WSGI)

## 🚧 Future Enhancements

- [ ] PDF/DOCX file upload support
- [ ] Advanced visualization of embeddings (t-SNE, UMAP)
- [ ] Multiple document support for comparison
- [ ] Custom chunking strategies
- [ ] Performance metrics and benchmarking
- [ ] API endpoints for external integration
- [ ] Docker containerization
- [ ] Cloud deployment guides

## 🤝 Contributing

Contributions are welcome! This project is designed to be educational, so clear explanations and well-documented code are highly valued.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear documentation
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 Python guidelines
- Include docstrings for all functions
- Add type hints where appropriate
- Write clear, educational comments

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- OpenAI for providing the embedding and language model APIs
- The open-source community for the libraries and tools used
- Educational resources that inspired the RAG implementation approach

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join conversations about RAG concepts and improvements
- **Documentation**: Check the code comments and inline explanations

---

**Happy Learning! 🎓** This project aims to make RAG concepts accessible and practical. Feel free to explore, experiment, and contribute to making AI more understandable for everyone.
