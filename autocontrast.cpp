//                                   ┌───────────────────────────────────────────┐
//                                   │    Copyright (c) 2022 Nikita Paniukhin    │
//                                   │      Licensed under the MIT license       │
//                                   └───────────────────────────────────────────┘
//
// =====================================================================================================================

// #pragma GCC optimize("Ofast")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,tune=native")
// #pragma GCC target("avx2")

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <omp.h>

using namespace std;



void handle_image(string input_path, string output_path, float coeff, bool debug=false) {
    if (debug) cout << "Handling \"" << input_path << "\"..." << endl;
    chrono::time_point<chrono::high_resolution_clock> start_time, end_time;

    #ifdef _OPENMP
        const int THREADS_COUNT = omp_get_max_threads();
    #else
        const int THREADS_COUNT = 1;
    #endif

    // ================================================ INITIALIZATION =================================================

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

    if (first_indentifier != 'P' || (second_indentifier != '5' && second_indentifier != '6')) {
        cout << "PNM file not recognized: \"P5\" or \"P6\" not found" << endl;
        return;
    }

    bool colored = (second_indentifier == '6');
    int size = width * height, colorwise_size = (colored ? 3 * size : size);

    if (debug) cout << "width: " << width << "\nheight: " << height << "\nsize: " << size << endl;


    if (debug) cout << "Allocating memory..." << endl;

    uint8_t *image = (uint8_t *) malloc(sizeof(uint8_t) * colorwise_size);
    if (!image) {
        cout << "Can not allocate memory for this file" << endl;
        return;
    }


    // ===================================================== INPUT =====================================================

    if (debug) {
        cout << "Reading file..." << endl;
        start_time = chrono::high_resolution_clock::now();
    }

    fread(image, 1, colorwise_size, input);

    if (debug) {
        end_time = chrono::high_resolution_clock::now();
        cout << "Read in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }

    fclose(input);



    // ================================================== PROCESSING ===================================================

    if (debug) cout << '\n' << "Processing..." << endl;
    start_time = chrono::high_resolution_clock::now();

    // ------------------------- Frequencies -------------------------
    int thread_block_size = colorwise_size / THREADS_COUNT;
    size_t freq[256] = {0};

    if (debug) start_time = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #ifdef _OPENMP
            int cur_thread_num = omp_get_thread_num();
        #else
            int cur_thread_num = 0;
        #endif

        int start = thread_block_size * cur_thread_num,
            end = thread_block_size * (cur_thread_num + 1);

        size_t tmp_freq[256] = {0};

        // #pragma omp parallel for
        for (int pixel_index = start; pixel_index < end; ++pixel_index) {
            ++tmp_freq[image[pixel_index]];
        }

        #pragma omp critical
        {
            for (int i = 0; i < 256; ++i) {
                freq[i] += tmp_freq[i];
            }
        }
    }

    if (debug) {
        end_time = chrono::high_resolution_clock::now();
        cout << "Frequences1 in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
        start_time = chrono::high_resolution_clock::now();
    }

    for (int pixel_index = thread_block_size * THREADS_COUNT; pixel_index < colorwise_size; ++pixel_index) {
        ++freq[image[pixel_index]];
    }

    if (debug) {
        end_time = chrono::high_resolution_clock::now();
        cout << "Frequences2 in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }

    // --------------------------- Borders ---------------------------
    int source_min, source_max;
    float needed_borders = coeff * size;

    if (debug) start_time = chrono::high_resolution_clock::now();

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            size_t pref_summ;
            for (pref_summ = 0, source_min = 0; source_min < 255; ++source_min) {
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
            for (pref_summ = 0, source_max = 255; source_max > 0; --source_max) {
                pref_summ += freq[source_max];
                if ((float) pref_summ > needed_borders) {
                    pref_summ -= freq[source_max];
                    break;
                }
            }
        }
    }


    if (debug) {
        end_time = chrono::high_resolution_clock::now();
        cout << "Borders in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
        cout << "min, max = " << (int) source_min << ' ' << (int) source_max << endl;
    }

    // ----------------------------- Processing ------------------------------
    float tmp = (float) 255.0 / (source_max - source_min);

    uint8_t mapping[256];
    for (int i = 0; i < 256; ++i) {
        mapping[i] = (uint8_t) min(255, (int) round(tmp * max(0, i - source_min)));
    }

    if (debug) start_time = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int pixel_index = 0; pixel_index < colorwise_size; ++pixel_index) {
        image[pixel_index] = mapping[image[pixel_index]];
    }

    end_time = chrono::high_resolution_clock::now();
    float elapsed = ((float) chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()) / 1000;
    printf("Time (%i thread(s)): %g ms\n", THREADS_COUNT, elapsed);


    // ==================================================== OUTPUT =====================================================

    if (debug) {
        cout << '\n' << "Writing output..." << endl;
        start_time = chrono::high_resolution_clock::now();
    }

    FILE * output = fopen(output_path.c_str(), "wb");
    fprintf(output, "P%d\n%d %d\n%d\n", (colored ? 6 : 5), width, height, color_space);
    fwrite(image, 1, colorwise_size, output);
    fclose(output);

    if (debug) {
        end_time = chrono::high_resolution_clock::now();
        cout << "Wrote in " << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms" << endl;
    }


    // ==================================================== THE END ====================================================

    free(image);
    if (debug) cout << '\n' << '\n' << endl;
}

int main(int argc, char* argv[]) {
    #ifdef _OPENMP
        omp_set_nested(1);
    #else
        cout << "Warning: OpenMP is turned off!" << endl;
    #endif

    // omp_set_num_threads(1);
    // handle_image("images/rgb.pnm", "result/rgb.pnm", 0, true);

    // omp_set_num_threads(72);
    // handle_image("images/picTest9.pnm", "result/picTest9.pnm", 0, true);

    // omp_set_num_threads(1);
    // handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);
    // for (int thread_cnt = 0; thread_cnt < 8; ++thread_cnt) {
    //     omp_set_num_threads(1 << thread_cnt);
    //     handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);
    // }
    // return 1;


    if (argc > 1) {
        if (argc < 5) {
            cout << "Too few arguments" << endl;
            return 1;
        }

        istringstream ss1(argv[1]);
        int threads_count;
        if (!(ss1 >> threads_count)) {
            cout << "Invalid number: " << argv[1] << endl;
            return 1;
        } else if (!ss1.eof()) {
            cout << "Trailing characters after number: " << argv[1] << endl;
            return 1;
        }

        istringstream ss2(argv[4]);
        float coeff;
        if (!(ss2 >> coeff)) {
            cout << "Invalid number: " << argv[4] << endl;
            return 1;
        } else if (!ss2.eof()) {
            cout << "Trailing characters after number: " << argv[4] << endl;
            return 1;
        }

        #ifdef _OPENMP
            omp_set_num_threads(threads_count);
        #endif
        handle_image(argv[2], argv[3], coeff);

    } else {
        cout << "No arguments specified, running with debug configuration..." << endl;

        handle_image("images/low_contrast.small.pnm", "result/low_contrast.small.pnm", 0.01, false);
        handle_image("images/low_contrast.large.pnm", "result/low_contrast.large.pnm", 0.01, false);
        handle_image("images/rgb.pnm", "result/rgb.pnm", 0, false);

        for (int i = 0; i <= 12; ++i) {
            if (i == 8) continue;
            handle_image(
                "images/picTest" + to_string(i) + ".pnm",
                "result/picTest" + to_string(i) + ".pnm",
                0
            );
        }
    }
}
