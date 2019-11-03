---
layout: post
title: Fast (and memory-safe) Fourier transforms in Rust
---

## The basics

MKL's [DFTI](https://software.intel.com/en-us/mkl-developer-reference-c-fft-functions) module has routines for performing Fast Fourier Transforms (FFTs) using thread-based parallelism, and seems to be a nice option to solve signal-processing problems in Rust projects, given how easy is to use the C FFI library. After generating [bindings](https://limads.github.io/2019/10/27/linking-against-mkl-in-rust-programs/) to the C module, it wasn't too hard to write a more idiomatic higher-level API that offers some compile-time guarantees. I'll describe the use and reasoning behind the API here, which should be applicable to other modules of the library.

The API of our binding will use the [ndarray::ArrayBase](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html) data structure, which is a container generic over several properties:

- Over the basic scalar type (`f32`, `f64`, and so on); 

- Over the ownership model (`Array` for an owned memory region, `ArrayView` for non-owned immutable references and `ArrayViewMut` for non-owned mutable references);

- Over the dimension of the data it holds (`Dim<usize>` for 1D arrays, `Dim<(usize, usize)`> for 2d arrays, and so on);

This generality allows to just plug-in a complex number such as `num::Complex<f32>` for the basic scalar type of the array that will store the forward output of the Fourier transform. The C interface does not work with complex number types, but rather with plain single-precision or double precision values, following the convention of representing complex numbers in their cartesian representation contiguously in memory:

![Complex number representation](img/complex-layout.png)

(The above layout is valid only for the `DFTI_CONJUGATE_EVEN_STORAGE = DFTI_COMPLEX_COMPLEX` setting of the DFTI descriptor)

Which, luckily, follows the same memory layout of `Array<Complex<D>>`, as long as we inform a numeric type D (either `f32` or `f64`) and dimension sizes that match the C array. MKL has an economical way of representing the output of the forward transform which is called conjugate-even storage, so called because every time we take the FFT of a real signal (which is usually the case when we are working with signals coming from actual measurements), the resulting transform will be even-symmetric (apart from the complex conjugate), so only the first portion of the signal relative to the point of symmetry needs to be represented. Since each element of the complex output now fills two scalar positions in memory, the halved output occupies a memory region which is almost the same size as the input.

## The API

We provide to the users a struct `FFTPlan` that follows the same generic design of `ndarray`. We make our `FFTPlan` generic with respect to the scalar type `A` and with respect to the dimension of the output `I`, so the user can detect any issues with respect to those parameters at compile time, and to reduce the amount of checking we have to do at runtime. After the struct is created, the user can repeatedly call the methods `.forward(data)` or `.backward(data)` to perform the forward and inverse transforms:

```rust
// Create FFTPlan
let n : usize = 100;
let mut fft_plan = FFTPlan::<f32, [Ix;1]>::new(
    [n as usize].into_dimension())?;

// Create test sin wave with angular frequency 4
let mut a = Array1::<f32>::linspace(0,(4.0)*2.0*PI,n);
a = a.map(|x| { x.sin() });

// Perform forward transform
match fft_plan.forward(a.view()) {
    Ok(b) => {
        // b is a ArrayViewMut<Complex<f32,Dim<[usize]> with the results.
	println!("Time domain: {:?}\n", a
	println!("Frequency domain: {:?}\n", 
		(1.0 / (n as f32))*to_amplitude_array(b.view()));
    },
    Err(s) => { println!("{}\n\n", s); }
}
```

The decision of returning an `ArrayViewMut` whose referenced `Array` is owned by `FFTPlan` avoids two things:

- Having to initialize a new memory region for each transform;

- The potentially costly copy to a container owned by the user. 

Since the calculated data is not re-used by FFTPlan in any other way (it just owns it), the user can modify it as needed without incurring the copy overhead. If the user does something that leaves the state of the underlying array invalid for the next transform, such as changing its dimensions, the next call to `.forward()` will simply return `Err(s)` informing a dimension size mismatch. Doing the whole manipulation inside a `match` or `if let` block not only guarantees the user is dealing with a valid output, but improves clarity of the program, since the lifetime of the mutable reference the user has access to is explicit.

If the user does not need to modify the array, he can take a `.view()` from the reference instead; or if he actually wants to own a copy of the result, he can call `.to_owned()` inside the block. 

Continuing with the example, the `to_amplitude_array()` is just a function to recover the amplitude information from the cartesian representation returned:

```rust
pub fn to_amplitude_array<A,D>(
    arr : ArrayView<Complex<A>,D>)
    -> Array<A, D>
    where D : Dimension,
    A : Float + Clone {

    arr.map(|a| { a.to_polar().0 })
}
```

After applying it to the array and normalizing it, we can visualize the content printed on the screen and check that the transform works:

![](/img/fft-results.png)

## Implementation

All information MKL needs to perform a FFT (dimension, precision, memory layout) is stored into a `DFTI_DESCRIPTOR` object, which we will wrap together with the owned memory regions that will contain the transform outputs:

```rust
pub struct FFTPlan<A, I : IntoDimension>
where Dim<I> : Dimension,
A : From<f32> + Copy + Zero {

    // Points to the owned descriptor. Valid for the lifetime of the struct.
    handle : *mut DFTI_DESCRIPTOR,

    // Owned memory region for the output of .forward()
    forward_buffer : Array<Complex<A>, Dim<I>>,

    // Owned memory region for the output of .backward()
    backward_buffer : Array<A, Dim<I>>,
}
```

In the MKL C API, all the information the `DFTI_DESCRIPTOR` requires is set at runtime, which is at odds with our design using generics. To bridge this gap, we write specialized implementation blocks for the kinds of transforms we need, that only contain a `new()` associated function, which will call the C API with the correct parameters encoded as literals on each implementation. This has the added benefit of only allowing the user to instantiate our object with types supported by the MKL (`f32` and `f64`). The example below instantiates a FFTPlan to solve the problem of 1D real Fourier transforms:

```rust
impl FFTPlan<f32, [Ix;1]> {
    pub fn new (
        input_dims : Dim<[Ix;1]>)
    -> Result<FFTPlan<f32, [Ix;1]>, String> {

        let mut backward_buffer =
            FFTPlan::<f32, [Ix;1]>::initialize_backward_buffer(input_dims);
        let mut forward_buffer =
            FFTPlan::<f32, [Ix;1]>::initialize_forward_buffer(input_dims);
        let sz0 = input_dims.into_pattern();

        // Encapsulates C API. Pass number of dimensions 
        //in tuple, and whether or not to use double precision.
        let handle = build_descriptor( (sz0, 0), false)?;
        Ok( FFTPlan::<f32, [Ix;1]>{ 
            handle, input_dims, forward_buffer, backward_buffer} )
    }
```

The `build_descriptor()` dispatches our arguments to the MKL C constants required at the descriptor initialization:

```rust
fn build_descriptor(dims : (usize, usize), double_prec : bool)
    -> Result<*mut DFTI_DESCRIPTOR, String> {

	/* (...) Transform our args to MKL constants */

    unsafe {
        let mut descriptor : DFTI_DESCRIPTOR =
            std::mem::uninitialized();
        let mut handle : *mut DFTI_DESCRIPTOR =
            &mut descriptor as *mut DFTI_DESCRIPTOR;
        let mut handle_ptr : *mut *mut DFTI_DESCRIPTOR =
            &mut handle as *mut *mut DFTI_DESCRIPTOR;
        let fail = DftiCreateDescriptor(
            handle,
            prec_const,
            prec_value,
            ndims as c_long,
            dims_arr
        ) as u32;
        check_dfti_status(fail as u32, "Could not construct Planner.")?;

        /* (...) Set remaining properties */

    }

    if fail == DFTI_NO_ERROR {
        Ok(handle)        
    }
```

Now that the correct descriptor is instantiated, the actual FFT computation can be generic over dimension and scalar type:

```rust
impl<A, I : IntoDimension> FFTPlan<A, I>
where Dim<I> : Dimension,
A : From<f32> + Copy + Zero {

    pub fn forward(&mut self, arr : ArrayView<A, Dim<I>>)
        -> Result<ArrayViewMut<Complex<A>, Dim<I>>, String>

        /* A runtime dimension check is required because 
        we opted for our users to be able to manipulate mutable references
        to the underlying owned buffer for performance reasons - 
        But our API is still safe because it will return an Error on 
        the event of dimension mismatch. */
        self.check_valid_forward_dim(arr.raw_dim())?;

        /* (...) Get pointers to input and buffer arrays via ArrayBase::mem_ptr() */
        unsafe {
            let status = DftiComputeForward(
                self.handle, in_ptr, out_ptr);
            if status == 0 {
                return Ok(self.forward_buffer.view_mut());
            } else {
                check_dfti_status(status as u32, "Error computing DFT.")?;
            }
        }
    }

    /* (...) Auxiliary bounds-checking functions, buffer initialization, etc. */
}
```

As soon as I get some details right, I plan to make this available on [crates.io](crates.io). Adding to this FFT API another set of routines that calls to the MKL convolution operations is all it takes for a production-ready crate with basic signal and image processing routines that benefit from the MKL performance and the memory safety afforded by the Rust compiler.

## Relevant links:

[Fast Fourier Transform (wiki)](https://en.wikipedia.org/wiki/Fast_Fourier_transform)

[MKL FFT functions documentation](https://software.intel.com/en-us/mkl-developer-reference-c-fft-functions)

[ndarray repository](https://github.com/rust-ndarray/ndarray)
