# ICA-filter-design-for-EEG-signals-using-C-
this repository is for the ica implementation of the EEG signal using CPP.
Description:

Summary:
This project focuses on developing a complete EEG signal processing workflow for the KaraOne dataset, combining C++ software development with hardware-accelerated deployment on Xilinx FPGAs using Vitis AI. The primary goal was to design a robust and extensible platform for real-time EEG signal preprocessing and artifact removal.

C++ Platform Development: Implemented Independent Component Analysis (ICA) for multi-channel EEG signals to remove artifacts such as eye-blinks, muscle activity, and line noise. Optimized computational efficiency and memory usage to handle large EEG datasets in real-time. The C++ platform served as the core signal processing engine, ensuring accurate preprocessing for downstream analysis.

Vitis AI Application Platform: Extended the C++ processing pipeline to a Vitis AI-based application, enabling hardware acceleration on Xilinx FPGAs (e.g., ZCU104). The platform design allowed seamless integration of software simulation with FPGA deployment, achieving faster processing and low-latency real-time performance.

Extensibility and Validation: Designed the system for modularity, allowing the C++ software platform to be easily scaled or updated for other EEG datasets or FPGA architectures. Validated the pipeline using the KaraOne dataset, confirming effective artifact removal and signal integrity for subsequent feature extraction and classification tasks.

Skills and Technologies: C++, Vitis AI, FPGA deployment, ICA, EEG signal processing, KaraOne dataset, real-time filtering, hardware-software co-design, embedded system integration.

Impact:
This project demonstrates a full-stack EEG signal processing solution from algorithmic development to hardware acceleration, providing a reliable platform for research and real-time BCI applications. The integration with Vitis AI ensures that the platform is future-ready for deployment on FPGA-based embedded systems.
.![ica_CPP](https://github.com/user-attachments/assets/943feb99-5584-4caa-84ef-f1fc8fd56ee9)
