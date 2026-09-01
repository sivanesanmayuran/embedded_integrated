#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "api.h"          /* PQClean ML-KEM-768 (clean) API */
#include "tflm_example.h" /* TFLM_Init / TFLM_Run */

#include "dsp/fir_filter.h"
#include "dsp/iir_filter.h"


// #define NUM_CHANNELS 4
// #define BLOCK_SIZE   4
// #define PI           3.14159265358979323846

// // Structure to hold complex numbers
// typedef struct {
//     double real;
//     double imag;
// } Complex;

// // Concept prototype for a lowpass prototype filter (Window function)
// // In a true polyphase filter bank, this would be a full FIR prototype filter.
// const double PROTOTYPE_FILTER[BLOCK_SIZE] = {0.35, 0.65, 0.65, 0.35};

// // Simplified 4-point DFT acting as our channelizer backend
// void discrete_fourier_transform(Complex* input, Complex* output) {
//     for (int k = 0; k < NUM_CHANNELS; k++) {
//         output[k].real = 0.0;
//         output[k].imag = 0.0;
        
//         for (int n = 0; n < BLOCK_SIZE; n++) {
//             double angle = 2 * PI * k * n / NUM_CHANNELS;
//             double cos_val = cos(angle);
//             double sin_val = sin(angle);
            
//             // Multiply input by e^(-j * 2 * pi * k * n / N)
//             output[k].real += input[n].real * cos_val + input[n].imag * sin_val;
//             output[k].imag += input[n].imag * cos_val - input[n].real * sin_val;
//         }
//     }
// }

// void channelizer() {
//     // Simulated input wideband data stream (4 complex samples)
//     Complex input_signal[BLOCK_SIZE] = {
//         {1.0, 0.0},
//         {0.707, 0.707},
//         {0.0, 1.0},
//         {-0.707, 0.707}
//     };
    
//     Complex filtered_signal[BLOCK_SIZE];
//     Complex channel_outputs[NUM_CHANNELS];

//     printf("=== Input Wideband Signal ===\n");
//     for (int i = 0; i < BLOCK_SIZE; i++) {
//         printf("Sample %d: Real = %6.3f, Imag = %6.3f\n", i, input_signal[i].real, input_signal[i].imag);
//     }

//     // Step 1: Apply prototype lowpass filter weights (Windowing)
//     for (int i = 0; i < BLOCK_SIZE; i++) {
//         filtered_signal[i].real = input_signal[i].real * PROTOTYPE_FILTER[i];
//         filtered_signal[i].imag = input_signal[i].imag * PROTOTYPE_FILTER[i];
//     }

//     // Step 2: Channelize using the DFT engine
//     discrete_fourier_transform(filtered_signal, channel_outputs);

//     // Step 3: Print separate channel outputs
//     printf("\n=== Channelizer Output (Separated Channels) ===\n");
//     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
//         double magnitude = sqrt(channel_outputs[ch].real * channel_outputs[ch].real + 
//                                 channel_outputs[ch].imag * channel_outputs[ch].imag);
        
//         printf("Channel %d (Bin %d): Real = %6.3f, Imag = %6.3f | Mag = %6.3f\n", 
//                ch, ch, channel_outputs[ch].real, channel_outputs[ch].imag, magnitude);
//     }
// }


// #define NUM_CHANNELS 16
// #define PI           3.14159265358979323846

// typedef struct {
//     double real;
//     double imag;
// } Complex;

// // 16-APSK Constellation Structure (DVB-S2 Standard setup: 4 inner, 12 outer points)
// Complex APSK16_CONSTELLATION[16];

// // Initialize the 16-APSK Constellation mapping
// void init_16apsk_constellation() {
//     double r1 = 1.0;       // Inner ring radius
//     double r2 = 2.75;      // Outer ring radius
    
//     // 4 points on the inner ring
//     for (int i = 0; i < 4; i++) {
//         double angle = i * (2.0 * PI / 4.0) + (PI / 4.0);
//         APSK16_CONSTELLATION[i].real = r1 * cos(angle);
//         APSK16_CONSTELLATION[i].imag = r1 * sin(angle);
//     }
//     // 12 points on the outer ring
//     for (int i = 0; i < 12; i++) {
//         double angle = i * (2.0 * PI / 12.0) + (PI / 12.0);
//         APSK16_CONSTELLATION[4 + i].real = r2 * cos(angle);
//         APSK16_CONSTELLATION[4 + i].imag = r2 * sin(angle);
//     }
// }

// // 16-Point DFT acting as our 16-channel downconverting Channelizer
// void channelizer_16(Complex* input, Complex* output) {
//     for (int k = 0; k < NUM_CHANNELS; k++) {
//         output[k].real = 0.0;
//         output[k].imag = 0.0;
        
//         for (int n = 0; n < NUM_CHANNELS; n++) {
//             // Forward DFT performs downconversion: e^(-j * 2 * pi * k * n / N)
//             double angle = 2 * PI * k * n / NUM_CHANNELS;
//             double cos_val = cos(angle);
//             double sin_val = sin(angle);
            
//             output[k].real += input[n].real * cos_val + input[n].imag * sin_val;
//             output[k].imag += input[n].imag * cos_val - input[n].real * sin_val;
//         }
//         // Normalise DFT output
//         output[k].real /= NUM_CHANNELS;
//         output[k].imag /= NUM_CHANNELS;
//     }
// }

// void channelizer() {
//     init_16apsk_constellation();

//     // 1. Simulate baseband data symbols we want to transmit across different channels
//     // Let's place distinct 16-APSK symbols on Channel 3 and Channel 7
//     int ch3_symbol_idx = 5;  // Index 5 (Outer ring symbol)
//     int ch7_symbol_idx = 1;  // Index 1 (Inner ring symbol)
    
//     Complex tx_symbols[NUM_CHANNELS] = {0}; // All other channels are empty (0)
//     tx_symbols[3] = APSK16_CONSTELLATION[ch3_symbol_idx];
//     tx_symbols[7] = APSK16_CONSTELLATION[ch7_symbol_idx];

//     printf("=== [TX] Generated Baseband Symbols ===\n");
//     printf("Channel 3 target baseband: Real = %6.3f, Imag = %6.3f\n", tx_symbols[3].real, tx_symbols[3].imag);
//     printf("Channel 7 target baseband: Real = %6.3f, Imag = %6.3f\n\n", tx_symbols[7].real, tx_symbols[7].imag);

//     // 2. Synthesise wideband signal (Modulate baseband symbols onto multiple carrier frequencies)
//     // This is mathematically done using an IDFT (Inverse DFT)
//     Complex wideband_signal[NUM_CHANNELS] = {0};
//     for (int n = 0; n < NUM_CHANNELS; n++) {
//         for (int k = 0; k < NUM_CHANNELS; k++) {
//             double angle = 2 * PI * k * n / NUM_CHANNELS;
//             wideband_signal[n].real += tx_symbols[k].real * cos(angle) - tx_symbols[k].imag * sin(angle);
//             wideband_signal[n].imag += tx_symbols[k].real * sin(angle) + tx_symbols[k].imag * cos(angle);
//         }
//     }

//     // 3. Process wideband signal through the Receiver Channelizer
//     // The channelizer automatically strips the respective carrier frequencies away
//     Complex rx_baseband_channels[NUM_CHANNELS];
//     channelizer_16(wideband_signal, rx_baseband_channels);

//     // 4. Verify recovered baseband signals
//     printf("=== [RX] Recovered Baseband Channels via Channelizer ===\n");
//     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
//         // Only print channels where we expect data to keep terminal tidy
//         if (ch == 3 || ch == 7 || fabs(rx_baseband_channels[ch].real) > 0.01) {
//             printf("Channel %2d: Real = %6.3f, Imag = %6.3f\n", 
//                    ch, rx_baseband_channels[ch].real, rx_baseband_channels[ch].imag);
//         }
//     }

// }


#define NUM_CHANNELS  16
#define TAPS_PER_PATH  4   // Total filter length = 16 * 4 = 64
#define FILTER_LEN     (NUM_CHANNELS * TAPS_PER_PATH)
#define PI            3.14159265358979323846

typedef struct {
    double real;
    double imag;
} Complex;

Complex APSK16_CONSTELLATION[16];
double PROTOTYPE_LPF[FILTER_LEN];
Complex polyphase_buffer[NUM_CHANNELS][TAPS_PER_PATH] = {0};

// 1. Initialize the 16-APSK Constellation mapping (DVB-S2 Standard setup)
void init_16apsk_constellation() {
    double r1 = 1.0;       // Inner ring radius
    double r2 = 2.75;      // Outer ring radius
    
    for (int i = 0; i < 4; i++) {
        double angle = i * (2.0 * PI / 4.0) + (PI / 4.0);
        APSK16_CONSTELLATION[i].real = r1 * cos(angle);
        APSK16_CONSTELLATION[i].imag = r1 * sin(angle);
    }
    for (int i = 0; i < 12; i++) {
        double angle = i * (2.0 * PI / 12.0) + (PI / 12.0);
        APSK16_CONSTELLATION[4 + i].real = r2 * cos(angle);
        APSK16_CONSTELLATION[4 + i].imag = r2 * sin(angle);
    }
}

// 2. Initialize a normalized Prototype Low-Pass Filter
void init_prototype_filter() {
    double sum = 0.0;
    for (int i = 0; i < FILTER_LEN; i++) {
        double t = i - (FILTER_LEN - 1) / 2.0;
        if (t == 0) PROTOTYPE_LPF[i] = 1.0;
        else PROTOTYPE_LPF[i] = sin(PI * 0.0625 * t) / (PI * 0.0625 * t); 
        
        double window = 0.5 * (1.0 - cos(2.0 * PI * i / (FILTER_LEN - 1)));
        PROTOTYPE_LPF[i] *= window;
        sum += PROTOTYPE_LPF[i];
    }
    // Normalize filter gain for the PFB structure
    for (int i = 0; i < FILTER_LEN; i++) {
        PROTOTYPE_LPF[i] *= (NUM_CHANNELS / sum);
    }
}

// 3. Multi-carrier Modulator Backend (IDFT)
void multi_carrier_modulate(Complex* tx_channels, Complex* wideband_out) {
    for (int n = 0; n < NUM_CHANNELS; n++) {
        wideband_out[n].real = 0.0;
        wideband_out[n].imag = 0.0;
        for (int k = 0; k < NUM_CHANNELS; k++) {
            double angle = 2 * PI * k * n / NUM_CHANNELS;
            wideband_out[n].real += tx_channels[k].real * cos(angle) - tx_channels[k].imag * sin(angle);
            wideband_out[n].imag += tx_channels[k].real * sin(angle) + tx_channels[k].imag * cos(angle);
        }
    }
}

// 4. Standard 16-point DFT Engine
void dft_16(Complex* input, Complex* output) {
    for (int k = 0; k < NUM_CHANNELS; k++) {
        output[k].real = 0.0;
        output[k].imag = 0.0;
        for (int n = 0; n < NUM_CHANNELS; n++) {
            double angle = 2 * PI * k * n / NUM_CHANNELS;
            output[k].real += input[n].real * cos(angle) + input[n].imag * sin(angle);
            output[k].imag += input[n].imag * cos(angle) - input[n].real * sin(angle);
        }
        output[k].real /= NUM_CHANNELS;
        output[k].imag /= NUM_CHANNELS;
    }
}

// 5. Polyphase Filter Bank Processing Engine
void polyphase_channelizer(Complex* wideband_in, Complex* rx_baseband_out) {
    Complex dft_input[NUM_CHANNELS] = {0};

    // Commutator step: Shift new wideband sample block into polyphase paths
    for (int p = 0; p < NUM_CHANNELS; p++) {
        for (int t = TAPS_PER_PATH - 1; t > 0; t--) {
            polyphase_buffer[p][t] = polyphase_buffer[p][t - 1];
        }
        polyphase_buffer[p][0] = wideband_in[p];
    }

    // Polyphase FIR filter step
    for (int p = 0; p < NUM_CHANNELS; p++) {
        for (int t = 0; t < TAPS_PER_PATH; t++) {
            int filter_idx = t * NUM_CHANNELS + p; 
            double h = PROTOTYPE_LPF[filter_idx];
            
            dft_input[p].real += polyphase_buffer[p][t].real * h;
            dft_input[p].imag += polyphase_buffer[p][t].imag * h;
        }
    }

    // Demodulation backend matrix
    dft_16(dft_input, rx_baseband_out);
}

int channelizer() {
    init_16apsk_constellation();
    init_prototype_filter();

    // Set up distinct 16-APSK symbols on Channel 2 and Channel 5
    Complex tx_channels[NUM_CHANNELS] = {0}; 
    tx_channels[2] = APSK16_CONSTELLATION[3];  // Inner ring point
    tx_channels[5] = APSK16_CONSTELLATION[12]; // Outer ring point

    printf("=== [TX] Transmitted Baseband 16-APSK Symbols ===\n");
    printf("Channel 2 target: Real = %6.3f, Imag = %6.3f\n", tx_channels[2].real, tx_channels[2].imag);
    printf("Channel 5 target: Real = %6.3f, Imag = %6.3f\n\n", tx_channels[5].real, tx_channels[5].imag);

    // Synthesize the multi-carrier wideband block
    Complex wideband_block[NUM_CHANNELS];
    multi_carrier_modulate(tx_channels, wideband_block);

    // Stream 5 consecutive identical blocks to fill the PFB filter memory lines (settling time)
    Complex rx_baseband[NUM_CHANNELS];
    for (int block = 1; block <= 5; block++) {
        polyphase_channelizer(wideband_block, rx_baseband);
        printf("--- Processing Streaming Data Block %d/5 ---\n", block);
    }

    // Print final recovered channels after the polyphase filter has fully settled
    printf("\n=== [RX] Recovered Baseband Channels (Steady State) ===\n");
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
        double mag = sqrt(rx_baseband[ch].real * rx_baseband[ch].real + rx_baseband[ch].imag * rx_baseband[ch].imag);
        if (mag > 0.1) { // Filter out empty noise channels
            printf("Channel %2d: Real = %6.3f, Imag = %6.3f (Mag = %6.3f)\n", 
                   ch, rx_baseband[ch].real, rx_baseband[ch].imag, mag);
        }
    }

    return 0;
}

void dsp_tests(void)
{
    FIRFilter my_fir;
    /* 16-tap low-pass FIR (Hamming-windowed sinc, cutoff ~0.25 x Nyquist).
       Symmetric (linear phase), normalized so the coefficients sum to 1.0. */
    static const float my_coeff[16] = {
        -0.001542f, -0.004302f, -0.006395f,  0.000000f,
         0.025493f,  0.075846f,  0.135615f,  0.175285f,
         0.175285f,  0.135615f,  0.075846f,  0.025493f,
         0.000000f, -0.006395f, -0.004302f, -0.001542f
    };

    fir_init(&my_fir, my_coeff, 16);
    fir_reset(&my_fir);

    /* Feed a short unit-step and print the filter response. */
    printf("Hey Lets run the DSP FIR test\n");
    for (int n = 0; n < 8; n++) {
        float y = fir_update(&my_fir, 1.0f);
        printf("  fir_update[%d] = %f\n", n, y);
    }

    /* IIR biquad demo: 2nd-order low-pass (fc ~ 0.1 fs, Q ~ 0.707).
       Coefficients are already normalized by a0. */
    IIRFilter my_iir;
    iir_init(&my_iir,
             0.020083f, 0.040166f, 0.020083f,   /* b0, b1, b2 */
            -1.561018f, 0.641352f);             /* a1, a2     */
    iir_reset(&my_iir);

    printf("Hey Lets run the DSP IIR test\n");
    const float in[8]  = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float       out[8];
    iir_update_block(&my_iir, in, out, 8);
    for (int n = 0; n < 8; n++) {
        printf("  iir_update[%d] = %f\n", n, out[n]);
    }
}

void pqclean_test(void)

{
	printf("Hey Lets run the PQC\n");
    // 1. Allocate buffers
    uint8_t pk[1184];
    uint8_t sk[2400];
    uint8_t ct[1088];
    uint8_t ss[32];   // This will hold the randomly generated key
    uint8_t ss1[32];  // This will hold the decrypted random key

    // 2. Generate Keypair
    int val1 = PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(pk, sk);

    // 3. Encapsulate (This generates a random key inside 'ss' AND creates the ciphertext 'ct')
    int val2 = PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(ct, ss, pk);

    // 4. Decapsulate (This extracts that same random key from 'ct' into 'ss1')
    int val3 = PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(ss1, ct, sk);

    // 5. Compare the keys (They will now match!)
    if (memcmp(ss, ss1, 32) == 0) {
        printf("Success! Both sides share the exact same 32-byte random key.\n");
    } else {
        printf("Error! Keys do not match.\n");
    }
}

void main(void)
{
	printf("PQC\n");
	pqclean_test();
    channelizer();
	dsp_tests();
	TFLM_Init();
	float a = 1.0;
    float b;
    printf("Hey Lets run the TFLM Run\n");
    TFLM_Run(a, &b);
}