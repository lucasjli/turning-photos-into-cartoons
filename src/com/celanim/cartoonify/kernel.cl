// Guassian Blur kernel
__kernel void gaussianBlur(__global int *pixels, __global int *newPixels,
                           const int width, const int height) {

    __const int GAUSSIAN_FILTER[25] = {
        2,  4,  5,  4, 2,
        4,  9, 12,  9, 4,
        5, 12, 15, 12, 5,
        4,  9, 12,  9, 4,
        2,  4,  5,  4, 2
    };
    const float GAUSSIAN_SUM = 159.0f;

    // Get current pixel coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Boundary check
    if (x >= width || y >= height) return;

    // Initialize accumulators for RGB channels
    float redSum = 0.0f, greenSum = 0.0f, blueSum = 0.0f;

    // Apply 5x5 Gaussian filter
    for (int ky = -2; ky <= 2; ky++) {      // Vertical kernel offset
        for (int kx = -2; kx <= 2; kx++) {  // Horizontal kernel offset
            // Clamp pixel coordinates to image boundaries
            int px = clamp(x + kx, 0, width - 1);
            int py = clamp(y + ky, 0, height - 1);
            // Get source pixel value
            int pixel = pixels[py * width + px];

            // Extract RGB components
            int r = (pixel >> 16) & 0xFF;  // Red channel
            int g = (pixel >> 8) & 0xFF;   // Green channel
            int b = pixel & 0xFF;          // Blue channel

            // Calculate kernel index
            int kidx = (ky + 2) * 5 + (kx + 2);
            int weight = GAUSSIAN_FILTER[kidx];

            // Accumulate weighted color values
            redSum += (float)r * weight;
            greenSum += (float)g * weight;
            blueSum += (float)b * weight;
        }
    }

    // Normalize and round the results
    int red = (int)round(redSum / GAUSSIAN_SUM);
    int green = (int)round(greenSum / GAUSSIAN_SUM);
    int blue = (int)round(blueSum / GAUSSIAN_SUM);

    // Combine channels back into ARGB format and store result
    newPixels[y*width+x] = 0xFF000000 | (clamp(red,0,255)<<16) | (clamp(green,0,255)<<8) | clamp(blue,0,255);
}

__kernel void sobelEdgeDetect(__global int *pixels, __global int *newPixels,
                              const int width, const int height, const int edgeThreshold) {

    // Predefined 3x3 Sobel operators
    const int sobelVertical[9] = {-1, 0, 1,
                                  -2, 0, 2,
                                  -1, 0, 1};
    const int sobelHorizontal[9] = {1,  2,  1,
                                    0,  0,  0,
                                   -1, -2, -1};

    // Get current pixel coordinates
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;  // Boundary check - return if coordinates are out of bounds

    // Initialize gradient accumulators
    int redV = 0, greenV = 0, blueV = 0;  // Vertical gradients
    int redH = 0, greenH = 0, blueH = 0;  // Horizontal gradients

    // Apply 3x3 Sobel filter
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            // Clamp pixel coordinates to image boundaries
            int px = clamp(x + kx, 0, width - 1);
            int py = clamp(y + ky, 0, height - 1);
            // Get source pixel value (ARGB format)
            int pixel = pixels[py * width + px];

            / Extract RGB components
            int r = (pixel & 0x00FF0000) >> 16;  // Red channel (bits 16-23)
            int g = (pixel & 0x0000FF00) >> 8;   // Green channel (bits 8-15)
            int b = (pixel & 0x000000FF);        // Blue channel (bits 0-7)
            // Calculate kernel index (0-8)
            int kidx = (ky + 1) * 3 + (kx + 1);

            // Accumulate vertical gradients
            redV += r * sobelVertical[kidx];
            greenV += g * sobelVertical[kidx];
            blueV += b * sobelVertical[kidx];

            // Accumulate horizontal gradients
            redH += r * sobelHorizontal[kidx];
            greenH += g * sobelHorizontal[kidx];
            blueH += b * sobelHorizontal[kidx];
        }
    }

    // Calculate absolute gradient magnitudes
    int vertGrad = abs(redV) + abs(greenV) + abs(blueV);  // Vertical edge strength
    int horizGrad = abs(redH) + abs(greenH) + abs(blueH); // Horizontal edge strength
    int totalGrad = vertGrad + horizGrad;

    // Thresholding - black for edges, white for non-edges
    // 0xFF000000 = opaque black (ARGB)
    // 0xFFFFFFFF = opaque white (ARGB)
    newPixels[y * width + x] = (totalGrad >= edgeThreshold) ? 0xFF000000 : 0xFFFFFFFF;
}


__kernel void reduceColours(__global int *oldPixels, __global int *newPixels,
		                    const int width, const int height, const int numColours) {

}

__kernel void mergeMask(__global int *maskPixels, __global int *photoPixels, __global int *newPixels,
		                const int maskColour, const int width) {

}

