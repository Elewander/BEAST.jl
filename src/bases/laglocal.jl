# T: coeff type
# Degree: degree
# Dim1: dimension of the support + 1
type LagrangeRefSpace{T,Degree,Dim1,NF} <: RefSpace{T,NF} end

numfunctions{T,D}(s::LagrangeRefSpace{T,D,2}) = D+1
numfunctions{T}(s::LagrangeRefSpace{T,0,3}) = 1
numfunctions{T}(s::LagrangeRefSpace{T,1,3}) = 3

valuetype{T}(ref::LagrangeRefSpace{T}, charttype) =
        SVector{numfunctions(ref), Tuple{T,T}}

# Evaluate constant lagrange elements on anything
(ϕ::LagrangeRefSpace{T,0}){T}(tp) = SVector(((one(T),zero(T)),))

# Evaluate linear Lagrange elements on a segment
function (f::LagrangeRefSpace{T,1,2}){T}(mp)
    u = mp.bary[1]
    j = jacobian(mp)
    SVector((u,-1/j), (1-u,1/j))
end

# Evaluete linear lagrange elements on a triangle
function (f::LagrangeRefSpace{T,1,3}){T}(t)
    u,v,w, = barycentric(t)
    SVector(u, v, w)
end


"""
    f(tangent_space, Val{:withcurl})

Compute the values of the shape functions together with their curl.
"""
function (f::LagrangeRefSpace{T,1,3}){T}(t, ::Type{Val{:withcurl}})
    # Evaluete linear Lagrange elements on a triange, together with their curl
    j = jacobian(t)
    u,v,w, = barycentric(t)
    p = t.patch
    SVector(
        (u, (p[3]-p[2])/j),
        (v, (p[1]-p[3])/j),
        (w, (p[2]-p[1])/j)
    )
end


# Evaluate constant Lagrange elements on a triangle, with their curls
function (f::LagrangeRefSpace{T,0,3}){T}(t, ::Type{Val{:withcurl}})
    i = one(T)
    z = zero(cartesian(t))
    (
        (i,z,),
    )
end


function curl(ref::LagrangeRefSpace, sh, el)
    sh1 = Shape(sh.cellid, mod1(sh.refid+1,3), -sh.coeff)
    sh2 = Shape(sh.cellid, mod1(sh.refid+2,3), +sh.coeff)
    return [sh1, sh2]
end


function strace(x::LagrangeRefSpace, cell, localid, face)

    Q = zeros(scalartype(x),2,3)

    p1 = neighborhood(face, 1)
    p2 = neighborhood(face, 0)

    u1 = carttobary(cell, cartesian(p1))
    u2 = carttobary(cell, cartesian(p2))

    P1 = neighborhood(cell, u1)
    P2 = neighborhood(cell, u2)

    vals1 = x(P1)
    vals2 = x(P2)

    for j in 1:numfunctions(x)
        Q[1,j] = vals1[j]
        Q[2,j] = vals2[j]
    end

    Q
end


function restrict{T}(refs::LagrangeRefSpace{T,0}, dom1, dom2)
    Q = eye(T, numfunctions(refs))
end

function restrict{T}(f::LagrangeRefSpace{T,1}, dom1, dom2)

    D = numfunctions(f)
    Q = zeros(T, D, D)

    # for each point of the new domain
    for i in 1:D
        v = dom2.vertices[i]

        # find the barycentric coordinates in dom1
        uvn = carttobary(dom1, v)

        # evaluate the shape functions in this point
        x = neighborhood(dom1, uvn)
        fx = f(x)

        for j in 1:D
            Q[j,i] = fx[j][1]
        end
    end

    return Q
end
