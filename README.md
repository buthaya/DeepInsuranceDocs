# DeepInsuranceDocs

**Deep Learning for Image Analysis in Insurance: Contextualized Reading**

Welcome to the DeepInsuranceDocs repository! This project focuses on advancing the field of insurance document analysis through the application of deep learning techniques, including models such as LayoutLM and Donut. Our goal is to enhance information retrieval and understanding from insurance documents.
DeepInsuranceDocs is a research project focused on my PhD thesis. This repository serves as a centralized hub for our research code, data, and documentation, designed to enhance information retrieval in insurance documents.

## Project Structure

Our project is organized into several key directories and components:

- **code**: This directory contains the core code for data preparation, model training, and evaluation. Subdirectories are organized according to the specific tasks and models.

   ```
   code/
   ├── data_preparation/
   ├── layoutlm_training/
   ├── donut_training/
   ├── ...
   ```

- **data**: Here, you can find sample datasets and data preprocessing scripts. Note that for larger datasets, instructions for obtaining the data may be provided.

   ```
   data/
   ├── dataset1/
   │   ├── train/
   │   ├── test/
   ├── dataset2/
   │   ├── ...
   ```

- **models**: Subdirectories represent different models, each containing model-specific code and configuration files.

   ```
   models/
   ├── LayoutLM/
   ├── Donut/
   ├── ...
   ```

- **utils**: Shared data preparation utilities, evaluation metrics, and other utility scripts can be found in this directory.

   ```
   utils/
   ├── data_preparation.py
   ├── evaluation_metrics.py
   ├── image_processing.py
   ```

- **config**: Configuration files for different models and data preparation settings are stored here.

   ```
   config/
   ├── layoutlm_config.json
   ├── donut_config.json
   ```

- **results**: This directory may include model checkpoints, evaluation results, and other output files.

   ```
   results/
   ├── layoutlm_model_checkpoint/
   ├── donut_model_checkpoint/
   ```

- **docs**: Documentation related to the research project, such as research papers, manuals, or project reports.

   ```
   docs/
   ├── research_paper.pdf
   ├── user_manual.pdf
   ```

- **notebooks**: Jupyter notebooks for data analysis, model training, and visualization are available in this directory.

   ```
   notebooks/
   ├── exploratory_data_analysis.ipynb
   ├── model_training_visualization.ipynb
   ```

## Usage

To use this repository effectively, follow these steps:

1. Clone the repository to your local machine:

   ```shell
   git clone https://github.com/YourUsername/DeepInsuranceDocs.git
   ```

2. Explore the code, data, and documentation in the respective directories.

3. Refer to the model-specific README files in each model directory for instructions on training and using the models.

4. To contribute or report issues, please follow our guidelines in the "Contributing" section below.

## Contributing

We welcome contributions from the research community. If you'd like to contribute, please:

- Fork the repository.
- Create a new branch for your work: `git checkout -b your-feature`
- Commit your changes and push to your fork.
- Submit a pull request to us. We will review your changes and collaborate.

## License

This project is open source and available under the [MIT License](LICENSE). Feel free to use and adapt the code and resources, following the terms of this license.

## Contact

For questions, feedback, or collaborations, please contact [me](mailto:uth.benno@gmail.com).

We look forward to advancing research in insurance document analysis together!
