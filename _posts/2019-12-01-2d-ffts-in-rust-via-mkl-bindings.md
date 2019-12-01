---
layout: post
title: 2D FFTs in Rust via MKL bindings
---

## Motivation

On the [previous post](https://limads.github.io/2019/11/03/ffts-in-rust/) I documented the early stage of writing a wrapper over [Intel MKL](https://software.intel.com/en-us/mkl) to perform FFTs of 1-dimensional arrays. In this post, we will expand this wrapper to work with 2D arrays (matrices). This data structure might be used to represent, for example, monochromatic images read from image files or captured from a camera in real time. High performance 2D FFTs such as the ones provided by MKL are required, for example, in tracking strategies such as [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation). Such bindings can be valuable in Rust programs to jumpstart a computer vision application with such requirements, or to offer easy access to a reference implementation against which Rust-native implementations can be tested.

## Memory layout of the 2D frequency domain

2D FFTs are slightly more challenging than the one-dimensional case because we have to consider a more complex output memory layout. Matrices with complex elements have to represent different things in each dimension: 

- Pairs of floating point elements for each entry in the first dimension (contiguous real and complex Cartesian components);

- Single floating point elements for each entry in the second dimension;

Following [MKL's documentation](https://software.intel.com/en-us/onemkl-developer-reference-c-fourier-transform-functions), we note that for the case of real-to-complex transforms we can use the conjugate-even storage strategy, so the memory space required for our output is always equal to the memory space required for our input, irrespective of the numbers of dimensions we have. This means that when building our Rust matrix data structure to store the output, we have to preserve one dimension of the input, and cut in half another dimension of the input (we just have to be mindful to adjust the MKL strides to respect which dimension was halved).

![img/ce2d.png](img/ce2d.png)

(Conjugate-even storage of FFT of an image with dimensions $m x n$)

## Using the nalgebra crate

Since the current binding crate is still in experimental stage, I figured I should experiment with another data structure for the wrapper. The [nalgbera](https://docs.rs/nalgebra/0.19.0/nalgebra/) crate exports a generic `Matrix` data structure, somewhat similar to `NDArray` in the sense that we can be generic over element type, but different from it in the sense that we can also be generic over the number of elements in each dimension (with the restriction of only being able to represent vectors or matrices). For our current application, this does not impose a serious limitation.

The benefit this crate offers for the current problem is a simpler API: we do not require a data structure to represent the dimensionality anymore, but can simply use the `DMatrix<N>` data structure for 2D arrays and the `DVector<N>` structure for 1D arrays. The D prefix identifies type aliases to structures whose number of elements is unknown at compile time (dynamic) in at least one dimension: `type DVector<N> = DMatrix<N,Dynamic,U1,C>` and `type DMatrix<N> = DMatrix<N,Dynamic,Dynamic,C>`. This simplification makes allocation, indexing and slicing slightly simpler. Another relative advantage is that this crate also offers a more self-contained set of features that are useful in a signal processing context, such as complex numbers and basic linear algebra operations.

## Implementation changes

Our `FFTPlan` data structure now has be generic over the dimensionality constraint types exported by `nalgebra`. We can have either a `FFTPlan<Dynamic, U1>` for 1D FFTs and `FFTPlan<Dynamic, Dynamic>` for 2D FFTs (the first dynamic type argument is added only for clarity). 

```rust
pub struct FFTPlan<N, R, C>
    where
        R : Dim,
        C : Dim,
        N : Scalar + Num + Clone + From<f32> + Copy + Debug,
        Complex<N> : Scalar + Num + Clone
{
    // Points to the owned descriptor. Valid for the lifetime of the struct.
    handle : *mut DFTI_DESCRIPTOR,

    // Holds the output of any the forward transform calls
    forward_buffer : Matrix<Complex<N>, Dynamic, C, VecStorage<Complex<N>, Dynamic, C>>,

    // Holds the output of any backward transform calls
    backward_buffer : Matrix<N, Dynamic, C, VecStorage<N, Dynamic, C>>,
}
```

When we had a single dimension, there was no need to consider the strides for the transform. Now that we have two dimensions, we need to inform MKL what the stride for each dimension is (how many elements to skip across dimension `k-1` so we can access the next element on dimension `k`. This is done by modifying the descriptor at our `build_descriptor()` call (see code of the previous post for reference):

```rust
let mut input_strides : [c_long;4] = [0,0,0,0];
let mut output_strides : [c_long;4] = [0,0,0,0];
DftiGetValue(
    handle,
    DFTI_CONFIG_PARAM_DFTI_INPUT_STRIDES,
    &input_strides as *const c_long
);
DftiGetValue(
    handle,
    DFTI_CONFIG_PARAM_DFTI_OUTPUT_STRIDES,
    &output_strides as *const c_long
);
if ndims >= 2 {
    // Halving the second dimension happens here
    output_strides [1] = input_strides[1] / 2 + 1;
    DftiSetValue(
        handle,
        DFTI_CONFIG_PARAM_DFTI_OUTPUT_STRIDES,
        &output_strides as *const c_long
    );
}
```

## Example

The following program allow us to create a small test pattern that varies over a horizontal spatial frequency parameter `u` and over the vertical spatial frequency parameter `v`. After applying the FFT, we save the original pattern and the magnitude of the spectrum as JPEG images.

```rust
use nsignal::fft::*;
use nalgebra::*;
use nsignal::io::*;
use std::f32::consts::PI;
use std::env;

fn main() {
    let n : usize = 16;
    let mut fft_plan = FFTPlan::<f32,Dynamic,Dynamic>::new((n,n)).unwrap();
    let args : Vec<usize> = env::args().skip(1).map(
        |a| a.parse::<usize>().expect("Args 1, 2 should be usize-compatible")
    ).collect();
    if args.len() < 2 {
        println!("Usage : fft2d [freq1] [freq2]");
        return;
    }
    let u = args[0] as f32;
    let v = args[1] as f32;
    let mut data : Vec<f32> = Vec::new();
        for x in 0..n {
            for y in 0..n {
                let x_dom = 2.0*PI*u*(x as f32 / n as f32);
                let y_dom = 2.0*PI*v*(y as f32 / n as f32);
                data.push( 0.0 + 1.0*(y_dom+x_dom).sin());
            }
        }
    let data = Matrix::from_data(VecStorage::new(
        Dynamic::new(n),Dynamic::new(n),data.into())
    );
    if let Ok(f) = fft_plan.forward(&data) {
        let ampl = to_amplitude_array(f);
        let ampl = 2.0*(1.0 / (n as f32).powf(2.0) )*ampl;
        matrix2image(&data, "src/examples/out/fft2d/space.jpg", 1.0, 0.5, true);
        if let Err(_) = matrix2image(&ampl, "src/examples/out/fft2d/freq.jpg", 0.0, 1.0, true) {
            println!("Error saving image!");
            return;
        }
    } else {
        println!("FFT error!");
        return;
    }
    println!("Done.");
}
```

![](img/2dfreqs.png)

