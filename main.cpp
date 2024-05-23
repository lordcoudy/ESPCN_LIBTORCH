#include <iostream>
#include <string>
#include <argparse/argparse.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <fstream>

int main(int argc, char* argv[]) {
    std::ofstream out;
    out.open("E:/SAVVA/STUDY/CUDA/ESPCN-Torch/results/espcn.csv", std::ios::app);

    argparse::ArgumentParser parser("Super Resolution Example");

    parser.add_argument("--input_image")
        .required()
        .help("input image to use");

    parser.add_argument("--model")
        .required()
        .help("model file to use");

    parser.add_argument("--output_filename")
        .default_value(std::string("output.png"))
        .help("where to save the output image");

    parser.add_argument("--cuda")
        .default_value(false)
        .implicit_value(true)
        .help("use cuda");

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        return EXIT_FAILURE;
    }

    auto input_image_path = parser.get<std::string>("--input_image");
    auto model_path = parser.get<std::string>("--model");
    auto output_filename = parser.get<std::string>("--output_filename");
    bool use_cuda = parser.get<bool>("--cuda");

    //out << "image, time, model\n";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // Load the image and convert to YCbCr
    cv::Mat img = cv::imread(input_image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return EXIT_FAILURE;
    }
    cv::Mat img_ycbcr;
    cv::cvtColor(img, img_ycbcr, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels(3);
    cv::split(img_ycbcr, channels);
    cv::Mat y = channels[0];
    cv::Mat cb = channels[1];
    cv::Mat cr = channels[2];

    // Load the model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
        if (use_cuda) {
            model.to(torch::kCUDA);
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return EXIT_FAILURE;
    }

    // Convert image to tensor
    y.convertTo(y, CV_32F, 1.0 / 255.0); // Normalize the Y channel to [0, 1]
    auto input_tensor = torch::from_blob(y.data, { 1, 1, y.rows, y.cols }, torch::kFloat);
    if (use_cuda) {
        input_tensor = input_tensor.to(torch::kCUDA);
    }

    // Run the model
    at::Tensor output_tensor = model.forward({ input_tensor }).toTensor();
    output_tensor = output_tensor.to(torch::kCPU);

    // Convert the output tensor to an image
    output_tensor = output_tensor.squeeze().detach();
    output_tensor = output_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    cv::Mat out_img_y(cv::Size(output_tensor.size(1), output_tensor.size(0)), CV_8U, output_tensor.data_ptr());

    // Resize Cb and Cr channels to the same size as Y channel
    cv::resize(cb, cb, out_img_y.size(), 0, 0, cv::INTER_LANCZOS4);
    cv::resize(cr, cr, out_img_y.size(), 0, 0, cv::INTER_LANCZOS4);

    // Merge channels back and convert to RGB
    std::vector<cv::Mat> out_channels = { out_img_y, cb, cr };
    cv::Mat out_img_ycbcr;
    cv::merge(out_channels, out_img_ycbcr);
    cv::Mat out_img;
    cv::cvtColor(out_img_ycbcr, out_img, cv::COLOR_YCrCb2BGR);

    // Save the output image
    cv::imwrite(output_filename, out_img);

    std::cout << "Output image saved to " << output_filename << std::endl;

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    out << input_image_path << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms, " << model_path << "\n";
    return 0;
}
