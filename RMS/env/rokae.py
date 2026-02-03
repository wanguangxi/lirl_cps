import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class ROKAE(DHRobot):
    """
    Class that models a Universal Robotics ROKAE manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``ROKAE()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.ROKAE()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_
        

    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        # robot length values (metres)
        a = [0, 0.03, 0.34, 0.035, 0, 0]
        d = [0.38, 0, 0, 0.345, 0, 0]

        alpha = [zero, -pi/2, zero, -pi/2, pi/2, -pi/2]

        # mass data, no inertia available
        mass = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        center_of_mass = [
                [0,      0.0,    0.00],
                [4.516,  0.472,  0.00],
                [0.108,  0.257,  0.04],
                [-0.077, -0.045, 0.00],
                [0.061,  0.060,  0.00],
                [-0.004, 0.024,  0.00]
            ]
        #dynamic - inertia of link with respect to COM
        I_ = [
                [ 0,         0,         3.063,         0,         0,         0],
                [ -1.159,    -20.399,  -18.347,       1.899,     -1.430,    0.052],
                [ -0.411,     -0.011,   2.135,         -0.096,    1.353,     0.248],
                [ -0.176,     -0.006,   -0.207,        0.075,    -0.082,    -0.066],
                [ -0.064,     -0.003,   0.057,         -0.015,    0.089,     -0.103],
                [ -0.022,     -0.000,   -0.045,        0.025,     -0.003,    0.011]
            ]
        Jm_ = [0, 0, 0.299, 0.447, 0.001, -0.181]
        #viscous friction
        B_ = [13.254, 58.605, 19.002, 4.826, 6.055, 2.360]
        # Coulomb friction
        Tc_ = [
                [ 10.029,       -10.029],
                [ 22.852,       -22.852],
                [ 10.957,       -10.957],
                [ 3.933,         -3.933],
                [ 2.879,         -2.879],
                [ 4.181,         -4.181]
             ]

        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                m=mass[j],
                r=center_of_mass[j],
                I = I_[j],
                Jm = Jm_[j],
                B = B_[j],
                Tc = Tc_[j],
                G=1

            )
            links.append(link)
    
        super().__init__(
            links,
            name="ROKAE",
            manufacturer="ROKAE Robotics",
            keywords=('dynamics', 'symbolic'),
            symbolic=symbolic
        )
    
        # zero angles
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))
        # horizontal along the x-axis
        self.addconfiguration("qr", np.r_[180, 0, 0, 0, 90, 0]*deg)

if __name__ == '__main__':    # pragma nocover

    ROKAE = ROKAE(symbolic=False)
    print(ROKAE)
    # print(ROKAE.dyntable())
    # print(ROKAE.qz)
    
