//                                   ┌───────────────────────────────────────────┐
//                                   │    Copyright (c) 2021 Nikita Paniukhin    │
//                                   │      Licensed under the MIT license       │
//                                   └───────────────────────────────────────────┘
//
// =====================================================================================================================

#pragma GCC optimize("Ofast")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,tune=native")
#pragma GCC target("avx2")

#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

// inline bool file_exists(const string& name) {
//   struct stat buffer;   
//   return (stat (name.c_str(), &buffer) == 0); 
// }

void handle_image(string input_path, string output_path, float coeff, bool debug=false) {
    if (debug) cout << "Handling \"" << input_path << "\"..." << endl;
    chrono::time_point<chrono::high_resolution_clock> start_time, end_time;
    const int THREADS_COUNT = omp_get_max_threads();

    // ================================================= INITIALIZATION ==================================================

    FILE * input = fopen(input_path.c_str(), "rb");

    if (!input) {
        cout << "Error reading input file!" << endl;
        return;
    }

    char first_indentifier, second_indentifier;
    int width, height, color_space;

    if (fscanf(input, "%c%c %d %d %d ", &first_indentifier, &second_indentifier, &width, &height, &color_space) != 5) {
        cout << "PNM file not recognized" << endl;
        return;
    }

    if (first_indentifier != 'P' || (second_indentifier != '6')) {
        cout << "PNM file not recognized: \"P5\" or \"P6\" not found" << endl;
        return;
    }

    int size = width * height;
    if (debug) cout << "width: " << width << "\nheight: " << height << "\nsize: " << size << endl;


    if (debug) cout << "Allocating memory..." << endl;

    // uint8_t image[height][width][3];
    // uint8_t *image = (uint8_t *) malloc(sizeof(uint8_t) * size * 3);
    
    uint8_t **image = (uint8_t **) malloc(sizeof(uint8_t*) * 3);
    uint8_t *raw_image = (uint8_t *) malloc(sizeof(uint8_t) * 3 * size);
    for (int color = 0; color < 3; ++color) image[color] = raw_image + size * color;

    if (!image || !raw_image) {
        cout << "Can not allocate memory for this file" << endl;
        return;
    }


    // ====================================================== INPUT ======================================================

    if (debug) {
        cout << "Reading file..." << endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

    char pixel_row[3 * width];
    for (int y = 0; y < height; ++y) {
        fread(pixel_row, 1, 3 * width, input);
        for (int x = 0; x < width; ++x) {
            for (int color = 0; color < 3; ++color) {
                image[color][y * width + x] = (uint8_t) pixel_row[x * 3 + color];
            }
        }
    }

    if (debug) {
        end_time = std::chrono::high_resolution_clock::now();
        cout << "Read in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }

    fclose(input);



    // =================================================== PROCESSING ====================================================

    if (debug) cout << '\n' << "Processing..." << endl;
    start_time = std::chrono::high_resolution_clock::now();

    // ------------------------- Frequencies -------------------------
    // int freq_threads = THREADS_COUNT, thread_block_size = (float) size / freq_threads;
    int freq_threads = THREADS_COUNT, thread_block_size = (size * 3) / THREADS_COUNT;
    size_t freq[256] = {0};

    if (debug) start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int cur_thread_num = omp_get_thread_num(),
            start = thread_block_size * cur_thread_num,
            end = thread_block_size * (cur_thread_num + 1);

        size_t tmp_freq[256] = {0};

        // #pragma omp parallel for
        for (int pixel_index = start; pixel_index < end; ++pixel_index) {
            ++tmp_freq[raw_image[pixel_index]];
        }

        #pragma omp critical
        {
            for (int i = 0; i < 256; ++i) {
                freq[i] += tmp_freq[i];
            }
        }
    }

    if (debug) {
        end_time = std::chrono::high_resolution_clock::now();
        cout << "Frequences1 in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

    for (int pixel_index = thread_block_size * freq_threads; pixel_index < size * 3; ++pixel_index) {
        for (int color = 0; color < 3; ++color) {
            ++freq[raw_image[pixel_index]];
        }
    }

    if (debug) {
        end_time = std::chrono::high_resolution_clock::now();
        cout << "Frequences2 in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }

    // --------------------------- Borders ---------------------------
    int source_min, source_max;
    float needed_borders = coeff * size;

    if (debug) start_time = std::chrono::high_resolution_clock::now();

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            size_t pref_summ;
            for (pref_summ = 0, source_min = 0; source_min < 256; ++source_min) {
                pref_summ += freq[source_min];
                if ((float) pref_summ > needed_borders) {
                    pref_summ -= freq[source_min];
                    break;
                }
            }
        }

        // #pragma omp section
        {
            size_t pref_summ;
            for (pref_summ = 0, source_max = 255; source_max >= 0; --source_max) {
                pref_summ += freq[source_max];
                if ((float) pref_summ > needed_borders) {
                    pref_summ -= freq[source_max];
                    break;
                }
            }
        }
    }


    if (debug) {
        end_time = std::chrono::high_resolution_clock::now();
        cout << "Borders in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
        cout << "min, max = " << (int) source_min << ' ' << (int) source_max << endl;
    }

    // ----------------------------- Processing ------------------------------
    float tmp = (float) 255.0 / (source_max - source_min), s_min = source_min;

    uint8_t mapping[256];
    for (int i = 0; i < 256; ++i) {
        mapping[i] = min((float) 255.0, max((float) 0.0, i - s_min) * tmp);
    }

    if (debug) start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int pixel_index = 0; pixel_index < size; ++pixel_index) {
        raw_image[pixel_index] = mapping[raw_image[pixel_index]];
    }

    // Old processing
    // #pragma omp parallel num_threads(72)
    // {
    //     #pragma omp for
    //     for (int pixel_index = 0; pixel_index < size; ++pixel_index) {
    //         for (int color = 0; color < 3; ++color) {
    //             image[color][pixel_index] = min(255.0, max(0.0, image[color][pixel_index] - s_min) * tmp);
    //         }
    //     }
    // }

    end_time = std::chrono::high_resolution_clock::now();
    float elapsed = ((float) chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()) / 1000;
    // cout << "Processed in " << elapsed << "us" << endl;
    printf("Time (%i thread(s)): %g ms\n", THREADS_COUNT, elapsed);


    // ====================================================== OUTPUT ======================================================

    if (debug) {
        cout << '\n' << "Writing output..." << endl;
        start_time = std::chrono::high_resolution_clock::now();
    }

    // Old output
    // fstream output(output_path, ios::out | ios::binary);
    // output << "P6" << '\n' << width << ' ' << height << '\n' << color_space << '\n';
    // for (int pixel_index = 0; pixel_index < size; ++pixel_index) {
    //     for (int i = 0; i < 3; ++i) {
    //         output << (char) image[i][pixel_index];
    //     }
    // }
    // output.close();

    FILE * output = fopen(output_path.c_str(), "wb");

    fprintf(output, "P6\n%d %d\n%d\n", width, height, color_space);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int color = 0; color < 3; ++color) {
                pixel_row[x * 3 + color] = (char) image[color][y * width + x];
            }
        }
        fwrite(pixel_row, 1, 3 * width, output);
    }

    fclose(output);

    if (debug) {
        end_time = std::chrono::high_resolution_clock::now();
        cout << "Wrote in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }

    // ===================================================== THE END ======================================================

    free(image);
    free(raw_image);

    if (debug) {
        cout << '\n' << '\n' << endl;
    }
}

int main(int argc, char* argv[]) {
    omp_set_nested(1);

    // omp_set_num_threads(72);
    // handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);

    // handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);

    // for (int thread_cnt = 0; thread_cnt < 8; ++thread_cnt) {
    //     omp_set_num_threads(1 << thread_cnt);
    //     handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);
    // }


    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            cout << argv[i] << endl;
        }

    } else {
        cout << "No arguments specified, running with debug configuration..." << endl;

        handle_image("images/low_contrast.small.pnm", "result/low_contrast.small.pnm", 0.01, false);
        handle_image("images/low_contrast.large.pnm", "result/low_contrast.large.pnm", 0.01, false);
        handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);

        for (int i = 0; i <= 12; ++i) {
            handle_image(
                "images/picTest" + to_string(i) + ".pnm",
                "result/picTest" + to_string(i) + ".pnm",
                0
            );
        }
    }
}
