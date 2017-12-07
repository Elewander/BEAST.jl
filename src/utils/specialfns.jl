export creategaussian
export derive
export integrate
export fouriertransform


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
