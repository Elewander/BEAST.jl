export creategaussian
export derive
export integrate
export fouriertransform
export createmodulatedgaussian

immutable Gaussian{T}
    scaling::T
    width::T
    delay::T
end


(g::Gaussian)(s::Real) = 4*g.scaling/(g.width*√π) * exp(-(4*(s-g.delay)/g.width)^2)


function creategaussian(width,s0,scaling=one(typeof(width)))
    #f(s) = 4*scaling/(width*sqrt(π)) * exp(-(4*(s-s0)/width)^2)
    Gaussian(scaling, width, s0)
    #f(s) = scaling * exp(-(4*(s-s0)/width)^2)
end


function  fouriertransform(g::Gaussian)
    scaling = g.scaling
    width = g.width
    s0 = g.delay
    ft(w) = scaling * exp(-im*w*s0 - (width*w/8)^2) / sqrt(2π)
end


"""
```math
    F(w) = 1/\sqrt{2pi} \int f(t) e^{-i ω t} dt
```
"""
function fouriertransform(a::Array, dt, t0, dim=1)
    n = size(a,dim)
    dω = 2π / (n*dt)
    b = fftshift(fft(a, dim), dim) * dt / sqrt(2π)
    ω0 = -dω * div(n,2)
    b, dω, ω0
end

"""
    integrate(f)

Returns a function that corresponds to a primitive of `f` whose value is 0 at -infinity.
```math
    F(t) = \int_{-\infty}{t} f(x) dx
```
"""
integrate(g::Gaussian) = s -> erfc(-4 * (s-g.delay)/g.width) / 2;

derive(g::Gaussian) =  s -> g(s) * (-8 * (s-g.delay)/g.width) * (4/g.width)

"""
```math
    f(t) = e^{-\frac{(t-t_0)^2}{2 \sigma^2}} cos(2 \pi f_0 (t-t_0))
```
"""
immutable ModulatedGaussian{T}
    scaling::T
    sigma::T
    delay::T
    frequency::T
end

"""
    createmodulatedgaussian(frequency, bandwidth, delay, amplitude)

Returns a modulated gaussian function where sigma = 6/(2*pi*bandwidth).
frequency: central frequency
bandwidth: frequency bandwidth
delay: time of peak amplitude
amplitude: peak amplitude
"""
function createmodulatedgaussian(frequency, bandwidth, delay, amplitude=one(typeof(frequency)))
	ModulatedGaussian(amplitude, 6/(2*pi*bandwidth), delay, frequency)
end

(mg::ModulatedGaussian)(s::Real) = exp(-(s-mg.delay)^2/(2*mg.sigma^2)) * cos(2*pi*mg.frequency*(s-mg.delay));
derive(mg::ModulatedGaussian) = s -> exp(-(s-mg.delay)^2/(2*mg.sigma^2)) * (-(s-mg.delay) / mg.sigma^2 * cos(2*pi*mg.frequency*(s-mg.delay)) - 2*pi*mg.frequency * sin(2*pi*mg.frequency*(s-mg.delay)));
integrate(mg::ModulatedGaussian) = s -> sqrt(pi/2) * mg.sigma * exp(-2*(pi*mg.frequency*mg.sigma)^2) * real(erfc(-((s-mg.delay) + 2*im*pi*mg.frequency*mg.sigma^2) / (sqrt(2)*mg.sigma)));
