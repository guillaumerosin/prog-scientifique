const global p = 1.23
const global R = 3.35e-2
const global m = 58e-3
const global g = 9.80665

isnull(v) = length(findall(iszero,v)) == length(v)
norm(v) = sqrt(v'v)

Cm(q) = isnull(q) ? 0 : 1.0 ./ (2.022 .+ 0.981 .* q)
