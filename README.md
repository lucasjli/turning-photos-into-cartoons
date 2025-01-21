# Assignment Three Turning Photos into Cartoons (20%)


**Due date 4 April at 11:30 pm**

## Instruction
* Fork the project by clicking the "Fork" button on the top-right corner.
* **Make sure that the visibility of your new repository (the one you just forked) is set to private.**
* You can obtain the URL of your forked project by clicking the "Clone" button (beside "Fork") and then the “copy to clipboard icon" at the right of the dropdown.
* Clone the new repository to your computer and open the project with IntelliJ, by using command line or IntelliJ interface
  * Using the Git command line: You will need to install a Git client yourself if you are not using the lab machines. In a termial on your computer, clone the assignment one repository to your computer using the command “git clone `<url you copied>`”. Open the project into your IntelliJ workspace. (File / Open the project directory).
  * IntelliJ: Alternatively, you could get the project from GitLab repository via IntelliJ interface. From the menu bar, 'Git' > 'Clone' > Copy the url to Repository URL > 'Clone' (Referenece: Get a project from version control: https://www.jetbrains.com/help/idea/import-project-or-module-wizard.html)
* Commit your changes regularly, providing an informative commit message and using Git inside IntelliJ (Commit and Changes Tutorial: https://www.jetbrains.com/help/idea/commit-and-push-changes.html)

You are expected to make at least 20 commits with messages to explain what have changed. `5 out of 60 marks are allocated for this`.


## Background
An animated movie production company called GelAnim wants to turn millions of photos (eg. from short video clips) into cartoon-style photos, so that they can incorporate them into the movies they are making. For example: 

|  Before Cartoonifying | After Cartoonifying |
|-----------|---------|
| <img src="https://elearn.waikato.ac.nz/pluginfile.php/3646570/mod_resource/content/1/eg_bumblebee2.jpg"/>|<img src="https://elearn.waikato.ac.nz/pluginfile.php/3646569/mod_resource/content/1/eg_bumblebee2_edited.jpg" />|

 	
GelAnim already have a Java program that does this, but it takes about FIVE seconds per 8 MPixel photo, which is far too slow. So they are asking you to see if you can port parts of the program to run on a GPU, so that it runs faster.


Their program uses two main techniques to turn an input photo into a cartoon-like photo:


* [Edge Detection](http://en.wikipedia.org/wiki/Edge_detection): They use the first couple of stages of the [Canny Edge Detector](http://en.wikipedia.org/wiki/Canny_edge_detector). This first blurs the image using a 5x5 Gaussian filter to smooth out the effects of any noisy pixels, and then uses a [Sobel filter](http://en.wikipedia.org/wiki/Sobel_operator) to detect edges in the photo. This gives a new image where all the edges are black and the other pixels are white;

* [Colour Quantization](http://en.wikipedia.org/wiki/Color_quantization): They use a very simple kind of colour quantization, which just rounds the colour values of each channel (red, green blue) down to just a few values. This gives a new image that uses only a few colours, and looks more like it is hand-painted;

* [Image Masking](http://en.wikipedia.org/wiki/Mask_(computing)): Finally, they use 'masking' to put the black edges on top of the quantized-colour photo, so that the final image has the edges outlined in black.

Note: this 'cartoonify' process is quite similar to the Cel Shading that is popular in some computer games, such as The Legend of Zelda: The Wind Waker.

The following sequence of images shows the output of each of the stages in GelAnim's program: 

<table style="width:100%">
  <tbody>
     <tr><th> Step </th><th>Result Image</th></tr>
<tr><td >1. The original image, showing just one small 200x200 area of the image. </td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3646571/mod_resource/content/1/eg_bumblebee.jpg" width="450"/></td></tr>
<tr><td >2. After applying the 5x5 Gaussian Blur filter. This takes each colour channel of each pixel and combines it with the values of the adjacent pixels, by multiplying them by the following matrix (so the pixel's own value is multiplied by 15, etc).
               
                [2,  4,  5,  4,  2]
                [4,  9, 12,  9,  4]
                [5, 12, 15, 12,  5]
                [4,  9, 12,  9,  4]
                [2,  4,  5,  4,  2]
                
</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3646572/mod_resource/content/1/eg_bumblebee_blurred.jpg"  /></td></tr>
<tr><td>
     3. After applying the horizontal and vertical Sobel edge filters to the blurred image:   
   
          vertical = [-1,  0, +1]    horizontal = [+1, +2, +1]
                     [-2,  0, +2]                 [ 0,  0,  0]
                     [-1,  0, +1]                 [-1, -2, -1]
  
                 
</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3646567/mod_resource/content/1/eg_bumblebee_edges.jpg" /></td></tr>
<tr><td >4. After applying the simple colour quantization algorithm to each pixel of the original image. This example uses just THREE values per channel, 
so all the red channel values are rounded to the nearest of 0, 127 or 255, and similarly for the green and blue channels.</td><td><img src="https://elearn.waikato.ac.nz/pluginfile.php/3646566/mod_page/content/1/eg_bumblebee_colours.jpg" /></td></tr>
<tr><td >5. The final 'cartoon' image is formed by drawing the black edges on top of the colour-reduced image. </td><td> <img src="https://elearn.waikato.ac.nz/pluginfile.php/3646568/mod_resource/content/1/eg_bumblebee_edited.jpg" /></td></tr>
    </tbody>
</table>

## Your Task
Your task is to implement a GPU version to their program, and make it run faster.

### GPU version of Cartoonify
Their GelAnim Cartoonify program has a command line interface with the following parameters. You should retain this usage, so that your program is backwards compatible with their existing workflow.

```
  Arguments:[-d] [-e EdgeThreshold] [-c NumColours] photo1.jpg photo2.jpg ...
    -d means turn on debugging, which saves intermediate photos.
    -e EdgeThreshold values can range from 0 (everything is an edge) up to about 1000 or more.
    -c NumColours is the number of discrete values within each colour channel (2..256).
```

A new "-g" flag before the existing "-d" flag is added for GPU. Eg.

```
  Arguments: [-g] [-d] [-e EdgeThreshold] [-c NumColours] photo1.jpg photo2.jpg ...
    -g use the GPU, to speed up photo processing.
    -d means turn on debugging, which saves intermediate photos.
    ...
```

If this "-g" flag is specified, then some or all of the photo processing steps will be done on the GPU, for faster performance. (Note: When the "-g" flag is specified, you can disable the debugging features if necessary, if they would slow down the speed of the program.) If the "-g" flag is omitted, then all the steps will continue to be done on the CPU using the existing code. So this original non-gpu program will be your benchmark, so that you compare the speed of your OpenCL programs against it to see how much speedup you have achieved.

### Short Report
As well as delivering the updated source code to CelAnim (as a complete IntelliJ project folder), you must also deliver a short report (i.e. a short .doc document) that tells the managers and technical personnel of CelAnim what you have achieved, and how you did it. It is suggested that your report should start with an executive summary that shows a graph that compares the original performance of the CelAnim program, and the improved performance of each program that you implemented.

The **body** of your report should briefly describe each of the new versions of the program that you implemented, and explain how much speedup (or slow-down) each one obtained, and why. This is so that they can learn from your experience, and see what approaches they should try in any other photo processing projects that they do in the future.

The summary page of your report should restate your achievements, mention any difficulties that you encountered, and give any suggestion for future improvements or approaches.

## Important notes

Impelement the *processPhotoOpenCL* method and several sub-functions in Cartoonify class that processes one input photo on GPU or GPU and CPU. 

*  Please make sure GPU and CPU implementation produce the same image outputs.
*  Please do not modify the main method or timing loops to change the time measurement statements.
*  Please note processPhotoOpenCL is the entry point of GPU implementation. It should contain the code (or methods) that initialize the environment and load the resources required by GPU version.

## Hints

You can run the original Cartoonify program (from within the project directory) from within IntelliJ, or on the command line like this:

```
            java -cp out/production/cartoonify com.celanim.cartoonify.Cartoonify -e 256 -c 3 img_examples/img_bucket.jpg
```

A typical usage with debugging on, would be:

```
            ./clean.sh   # to remove any old output files and stop them being inputs!
            java -cp out/production/cartoonify com.celanim.cartoonify.Cartoonify -d -e 256 -c 3 img_examples/img_bucket.jpg
```

Start by determining which methods are the bottleneck. Use Java profiling or VisualVM (https://visualvm.github.io/), and/or turn on debugging and inspect the timing of each stage. Read those methods until you understand them, and then think about how you could speed them up.

Processing pixels near the edge requires special case code, but we don't really care too much exactly what is output for a couple of rows/columns of pixels at the edge, since those parts of the output image will usually be discarded during further stages of the movie-making process.

Currently, the Cartoonify program reflects the edge of the image back onto itself. If you decide that more efficient techniques are possible, you could implement an alternative wrap method and use it in the convolution method. Just make sure that you test your approach, and measure its performance. If your changes make some of the existing unit tests fail, then either fix those tests to reflect your changes, or add the @Ignore(reason) annotation to the failing tests so that they are not run.

After you have sped up the Java code, it is time to move the bottleneck methods onto the GPU, by converting them into kernels. You have considerable freedom here to use a 1D or 2D grid, to choose different workgroup sizes, and to decide which methods in the processing pipeline should be done on the GPU.

When converting CPU process to GPU, draw the dependency graph (i.e. A Directed Acyclic Graph) between the four image processing methods, GaussianBlur, SobelEdgeDetect, ReduceColour, MergeMask. Try moving these methods to OpenCL kernels one by one, and after each code change re-run the Cartoonify program to check if any change is made to the output image. For example, you can run Gaussian Blur on GPUs and keep all other the processing methods on CPUs. Then check whether output images are correct (i.e. GPU and CPU output images should have the same md5sum hash value) before proceeding to the next method. md5sum is available on Lab one machines.
Could you reduce the overhead of memory allocation and memory copying from host to GPU by sharing memory objects?

Based on the dependency graph, could you use multiple queues to run kernels concurrently and use cl_events to set up synchronization points? 

## Submission Checklist

Make sure that your program runs correctly (i.e. no run-time exceptions) and GPU and CPU implementation produce the same image outputs.

```
            java -cp out/production/cartoonify com.celanim.cartoonify.Cartoonify -g -e 256 -c 3 img_examples/img_bucket.jpg img_examples/img_bumblebee.jpg img_examples/img_dog.jpg img_examples/img_pavlova.jpg img_examples/img_shenzhen.jpg img_examples/img_sunflower.jpg img_examples/img_surfers.jpg
     
```   

The output messages should look similar as follows.

```
Using edge threshold 256
Using 3 discrete colours per channel.
Done img_examples/img_bucket.jpg -> img_examples/img_bucket_cartoon.jpg in 2.924 secs.
Done img_examples/img_bumblebee.jpg -> img_examples/img_bumblebee_cartoon.jpg in 2.86 secs.
Done img_examples/img_dog.jpg -> img_examples/img_dog_cartoon.jpg in 2.93 secs.
Done img_examples/img_pavlova.jpg -> img_examples/img_pavlova_cartoon.jpg in 2.907 secs.
Done img_examples/img_shenzhen.jpg -> img_examples/img_shenzhen_cartoon.jpg in 2.844 secs.
Done img_examples/img_sunflower.jpg -> img_examples/img_sunflower_cartoon.jpg in 2.921 secs.
Done img_examples/img_surfers.jpg -> img_examples/img_surfers_cartoon.jpg in 1.107 secs.
Average processing time is 2.641 for 7 photos.
```

## Grading (60 marks in total)
|Marks | Allocated to |
|------|--------------|
|5    | At least twenty informative Commit comments |
|10    | Your GPU version runs correctly and produces the same results as the CPU version does (checked using md5sum)|
|10    | (a) Experiments and Graphs [8 marks], (b) Discussion of the difficulties that you encountered, and any suggestions for future improvements or approaches. [2 marks]|
|5     | Speed up the original java code (without GPU) |
|15    | GPU speedup |
|15    | OpenCL implementation (multiple-queues, memory objects sharing, etc.) and kernel optimizations|

