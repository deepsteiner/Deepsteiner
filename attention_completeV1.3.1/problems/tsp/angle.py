from math import sqrt, hypot, pi, cos, sin, atan2, tan
def st_point_generate(Ax, Ay, Bx, By, deg):
    c = hypot(By - Ay, Bx - Ax)
    b = c*cos(deg)
    inclinationAB = atan2(By - Ay, Bx - Ax)
    inclinationAC_p = inclinationAB + deg
    inclinationAC_n = inclinationAB - deg
    part = sqrt(3)/tan(deg)
    Dx1 = Ax + b * cos(inclinationAC_p) * (part - 1) / part
    Dy1 = Ay + b * sin(inclinationAC_p) * (part - 1) / part
    Dx2 = Ax + b * cos(inclinationAC_n) * (part - 1) / part
    Dy2 = Ay + b * sin(inclinationAC_n) * (part - 1) / part
    return Dx1, Dy1, Dx2, Dy2
