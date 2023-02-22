import numpy as np
import math
import cmath

class Energy:

    def __init__(self, x, y, depth, ship_properties, *args, **kwargs):
        self.x = x
        self.y = y
        self.depth = depth
        self.P = ship_properties["P"]
        self.T = ship_properties["T"]
        self.L = ship_properties["L"]
        self.B = ship_properties["B"]
        self.A = ship_properties["A"]
        self.P_installed = ship_properties["P_installed"]
        self.C_B = ship_properties["C_B"]
        self.x0 = ship_properties["x0"]
        self.N = ship_properties["N"]

        assert self.T < min(depth), "The draft cannot be larger than the water depth"
        assert len(x) == len(y) == len(depth), "the x, y coordinates and corresponding depth should have the same length"

    def HM(self, V, depth, angle):
        T = self.T
        L = self.L
        B = self.B
        A = self.A
        P_installed = self.P_installed * 1000
        C_B = self.C_B
        x0 = self.x0
    
            #Characteristics of ship
    
        T_F = T
        h_B = 0.6 * T_F
        rho = 1025
        g = 9.81
        v = 1 * 10**-6
        S_B = L * B

        C_M = 1.006 - 0.0056 * C_B**-3.56
        C_WP = (1 + 2 * C_B) / 3
        C_BB = 0.2
        A_BT = 0 #C_BB * B * T * C_M
        A_T = C_BB * B * T
        C_P = C_B / C_M
        lcb = -13.5 + 19.4 * C_P
        Delta = C_B * B * T * L
        x0 = 1 #number of screws
        P_installed = 25000000

            #Reynolds and Froude numbers:
    
        if V > 0:
            
            Re = V * L / v
            F_n = V / np.sqrt(g * L)
            F_nh = V / np.sqrt(g * depth)

                #Karpov

            if F_nh <= 0.4:
                if 0 <= (depth / T) < 1.75:
                    a_xx = (-4 * 10**(-12)) * F_nh**3 - 0.2143 * F_nh**2 - 0.0643 * F_nh + 0.9997
                elif 1.75 <= (depth / T) < 2.25:
                    a_xx = -0.8333 * F_nh**3 + 0.25 * F_nh**2 - 0.0167 * F_nh + 1
                elif 2.25 <= (depth / T) < 2.75:
                    a_xx = -1.25 * F_nh**4 + 0.5833 * F_nh**3 - 0.0375 * F_nh**2 - 0.0108 * F_nh + 1
                elif (depth / T) >= 2.75:
                    a_xx = 1
            
            if F_nh > 0.4:
                if 0 <= (depth / T) < 1.75:
                    a_xx = -0.9274 * F_nh**6 + 9.5953 * F_nh**5 - 37.197 * F_nh**4 + 69.666 * F_nh**3 - 65.391 * F_nh**2 + 28.025 * F_nh -3.4143
                elif 1.75 <= (depth / T) < 2.25:
                    a_xx = 2.2152 * F_nh**6 - 11.852 * F_nh**5 + 21.499 * F_nh**4 - 12.174 * F_nh**3 - 4.7873 * F_nh**2 + 5.8662 * F_nh - 0.2652
                elif 2.25 <= (depth / T) < 2.75:
                    a_xx = 1.2205 * F_nh**6 - 5.4999 * F_nh**5 + 5.7966 * F_nh**4 + 6.6491 * F_nh**3 - 16.123 * F_nh**2 + 9.2016 * F_nh - 0.6342
                elif 2.75 <= (depth / T) < 3.25:
                    a_xx = -0.4085 * F_nh**6 + 4.534 * F_nh**5 - 18.443 * F_nh**4 + 35.744 * F_nh**3 - 34.381 * F_nh**2 + 15.042 * F_nh - 1.3807
                elif 3.25 <= (depth / T) < 3.75:
                    a_xx = 0.4078 * F_nh**6 - 0.919 * F_nh**5 - 3.8292 * F_nh**4 + 15.738 * F_nh**3 - 19.766 * F_nh **2 + 9.7466 * F_nh - 0.6409
                elif 3.75 <= (depth / T) < 4.5:
                    a_xx = 0.3067 * F_nh**6 - 0.3404 * F_nh**5 - 5.0511 * F_nh**4 + 16.892 * F_nh**3 - 20.265 * F_nh**2 + 9.9002 *  F_nh - 0.6712
                elif 4.5 <= (depth / T) < 5.5:
                    a_xx = 0.3212 * F_nh**6 - 0.3559 * F_nh**5 - 5.1056 * F_nh**4 + 16.926 * F_nh**3 - 20.253 * F_nh**2 + 10.013 * F_nh - 0.7196
                elif 5.5 <= (depth / T) < 6.5:
                    a_xx = 0.9252 * F_nh**6 - 4.2574 * F_nh**5 + 5.0363 * F_nh**4 + 3.3282 * F_nh**3 - 10.367 * F_nh**2 + 6.3993 * F_nh - 0.2074
                elif 6.5 <= (depth / T) < 7.5:
                    a_xx = 0.8442 * F_nh**6 - 4.0261 * F_nh**5 + 5.313 * F_nh**4 + 1.6442 * F_nh**3 - 8.1848 * F_nh**2 + 5.3209 * F_nh - 0.0267
                elif 7.5 <= (depth / T) < 8.5:
                    a_xx = 0.1211 * F_nh**6 + 0.628 * F_nh**5 - 6.5106 * F_nh**4 + 16.7 * F_nh**3 - 18.267 * F_nh**2 + 8.7077 * F_nh - 0.4745
                elif 8.5 <= (depth / T) < 9.5:
                    if F_nh < 0.6:
                        a_xx = 1
                    elif F_nh >= 0.6:
                        a_xx = -6.4069 * F_nh**6 + 47.308 * F_nh**5 - 141.93 * F_nh**4 + 220.23 * F_nh**3 - 185.05 * F_nh**2 + 79.25 * F_nh - 12.484
                elif (depth / T) >= 9.5:
                    if F_nh < 0.6:
                        a_xx = 1
                    elif F_nh >= 0.6:
                        a_xx = -6.0727 * F_nh**6 + 44.97 * F_nh**5 -135.21 * F_nh**4 + 210.13 * F_nh**3 - 176.72 * F_nh**2 + 75.728 * F_nh - 11.893

                #Resistance due to friction

                #R_F: frictional resistance according to the ITTC-1957 friction formula 
                #C_F: frictional resistance coefficient dependent on the water depth

            C_Fdeep = 0.08169 / (math.log10(Re) - 1.717)**2
            C_Fproposed = (0.08169 / (math.log10(Re) - 1.717)**2) * (1 + (0.003998 / ((math.log10(Re) - 4.393))) * ((depth - T) / L)**-1.083)
            C_F0 = (0.075 / (((math.log10(Re)) - 2) ** 2))

            a = 0.042612 * math.log10(Re) + 0.56725
            C_Fkatsui = 0.0066577 / ((math.log10(Re)- 4.3762))**a

            V1 = 0.4277 * V * np.exp((depth/T)**-0.07634)

            S_T = L * (2 * T + B) * np.sqrt(C_M) * (0.453 + 0.4425 * C_B - 0.2862 * C_M - 0.003467 * (B / T) + 0.3696 * C_WP) + 2.38 * (A_BT / C_B)

            C_F = C_F0 + (C_Fproposed - C_Fkatsui) * (S_B / S_T) * (V1 / V)**2

            R_F = 0.5 * rho * C_F * S_T * V**2

            #1 + k1: form factor describing the viscous resistance of the hull form in relation to R_F
            c14 = 1 # + 0.011 * C_stern

            L_R = L * (1 - C_P + (0.06 * C_P * lcb)/(4 * C_P - 1))
            k1 = -0.07 + 0.487118 * c14 * (B / L)**1.06806 * (T / L)**0.46106 * (L / L_R)**0.121563 * (L**3 / Delta)**0.36486 * (1 - C_P)**-0.60247

                #R_APP: resistance of appendages

            R_APP = 0.5 * rho * C_F * 0.05 * S_T * V**2 * 2.5

            R_friction = R_F * (1 + k1) + R_APP
            
                #Resistance due to waves

            V2 = V / a_xx
            F_nV2 = V2 / np.sqrt(g * L)

            if (B / L) <= 0.11:
                c7 = 0.229577 * (B / L)**0.33333
            if 0.11 < (B / L) < 0.25:
                c7 = B / L
            if (B / L) >= 0.25:
                c7 = 0.5 - 0.0625 * (L / B)

            i_E = 1 + 89 * np.exp(-(L / B)**0.80856 * (1 - C_WP)**0.30484 * (1 - C_P - 0.0225 * lcb)**0.6367 * (L_R / B)**0.34574 * (100 * Delta / L**3)**0.16302)

            c1 = 2223105 * c7**3.78613 * (T / B)**1.07961 * (90 - i_E)**-1.37565

            c3 = 0.56 * A_BT**1.5 / (B * T * (0.31 * np.sqrt(A_BT) + T_F - h_B))
            c2 = np.exp(-1.89 * np.sqrt(c3))

            c5 = 1 - 0.8 * A_T / (B * T * C_M)

            if C_P <= 0.8:
                c16 = 8.07981 * C_P - 13.8673 * C_P**2 + 6.984388 * C_P**3
            elif C_P >= 0.8:
                c16 = 1.73014 - 0.7067 * C_P

            m1 = 0.0140407 * (L / T) - 1.75254 * Delta**(1/3) / L - 4.79323 * (B / L) - c16

            if L**3 / Delta <= 512:
                c15 = -1.69385
            elif 512 <= L**3 / Delta <= 1726.91:
                c15 = -1.69385 + ((L / Delta**(1/3)) - 8) / 2.36
            elif 1726.91 <= L**3 / Delta:
                c15 = 0

            m4 = c15 * 0.4 * np.exp(-0.034 * (F_nV2**-3.29))
                
            c17 = 6919 * C_M**(-1.3346) * (Delta / L**3)**2.00977 * ((L / B) - 2)**1.40692
            
            m3 = -7.2035 * (B / L)**0.326869 * (T / B)**0.605375

            if (L / B) <= 12:
                λ = 1.446 * C_P - 0.03 * (L / B)
            elif 12 <= (L / B):
                λ = 1.446 * C_P - 0.36

            if F_nV2 < 0.4:
                R_W1 = rho * g * Delta * c1 * c2 * c5 * np.exp(m1 * F_nV2**-0.9 + m4 * np.cos(λ * F_nV2**-2))
                R_W = R_W1
            
            elif F_nV2 > 0.55:
                R_W2 = rho * g * Delta * c17 * c2 * c5 * np.exp(m3 * F_nV2**-0.9 + m4 * np.cos(λ * F_nV2**-2))
                R_W = R_W2

            elif 0.4 <= F_nV2 < 0.55:
                R_W1 = rho * g * Delta * c1 * c2 * c5 * np.exp(m1 * F_nV2**-0.9 + m4 * np.cos(λ * F_nV2**-2))
                R_W2 = rho * g * Delta * c17 * c2 * c5 * np.exp(m3 * F_nV2**-0.9 + m4 * np.cos(λ * F_nV2**-2))
                R_W3 = R_W1 + ((10 * F_nV2 - 4) * (R_W2 - R_W1)) / 1.5
                R_W = R_W3

            #Additional Pressure Resistance of Bulbous Bow near the Water Surface

            P_B = 0.56 * np.sqrt(A_BT) / (T_F - 1.5*h_B)
            F_ni = V / np.sqrt(g * ( T_F - h_B - 0.25 * np.sqrt(A_BT)) + 0.15 * V**2)

            R_B = 0.11 * np.exp(-3 * P_B**-2) * F_ni**3 * A_BT**1.5 * rho * g / (1 + F_ni**2)
            
            #Additional Pressure Resistance of Immersed Transom Immersion

            F_nT = V / np.sqrt((2 * g * A_T) / (B + B * C_WP))

            if F_nT <= 5:
                c6 = 0.2 * (1 - 0.2 * F_nT)
            elif 5 <= F_nT:
                c6 = 0

            R_TR = 0.5 * rho * V**2 * A_T * c6
            
            #Residual resistance

            if (T_F / L) <= 0.04:
                c4 = T_F / L
            elif 0.04 < T_F / L:
                c4 = 0.04

            cA = 0.006 * (L + 100)**-0.16 - 0.00205 + 0.003 * np.sqrt(L / 7/5) * C_B**4 * c2 * (0.04 - c4)

            R_A = 0.5 * rho * V**2 * S_T * cA

            #Drag Resistance
            if angle > 0:
                R_Drag = self.Drag(V, angle)
            elif angle == 0:
                R_Drag = 0

            #Total ship resistance:

            R_HM = R_F * (1 + k1) + R_APP + R_W + R_B + R_TR + R_A 
            R_total = R_F * (1 + k1) + R_APP + R_W + R_B + R_TR + R_A + R_Drag
            
            #Brake horse power P_b
            
            P_e = V * R_total

            eta_0 = 0.6
            eta_r = 0.98

            D_s = 0.7 * T

            if F_n < 0.2:
                delta_w = 0
            elif F_n >= 0.2:
                delta_w = 0.1
                
            w = 0.11 * (0.16 / x0) * C_B * np.sqrt(Delta**(1 / 3) / D_s) - delta_w

            if x0 == 1:
                t = 0.6 * w * (1 + 0.67 * w)
            elif x0 == 2:
                t = 0.8 * w * (1 + 0.25 * w)

            eta_h = (1 - t) / (1 - w)

            P_d = P_e / (eta_0 * eta_r * eta_h)

            eta_t = 0.98
            eta_g = 0.96

            P_b = P_d / (eta_t * eta_g)

            #Power required for hotel

            P_h = 0.05 * P_installed

            #Total power required for the vessel

            P_total = P_b + P_h 

            return P_total, R_total, R_Drag, R_HM
        
        if V == 0:

            P_total = 0.05 * P_installed
            R_total = 0
            R_Drag = 0
            R_HM = 0

            return P_total, R_total, R_Drag, R_HM
    
    def Drag(self, V, angle):
        C_D_values = np.array([0, 0.09, 0.23, 0.45, 0.57])
        angle_values = np.array([0, 10, 20, 30, 35])
        aD, bD, cD = np.polyfit(angle_values, C_D_values, 2)

        C_D = aD * angle**2 + bD * angle + cD
        rho = 1000
        F_drag = 0.5 * C_D * self.A * 1000 * V**2

        return F_drag
    
    def solve_velocity_drift(self, *args, **kwargs):
        depth = self.depth
        x = self.x
        y = self.y

        velocity = []
        drift_angle = []
        speed_overground = []
        time = []

        if 'starting_time' in kwargs:
            t = kwargs['starting_time']

        if len(args) > 0:
            u = args[0]
            v = args[1]
            for i in range(len(depth) - 1):
                power = self.P
                start_velocity = 0
                end_velocity = 10
                tolerance = 10 ** -3

                if i > 0: 
                    t += ((x[i] - x[i-1]) * 1000) / self.speed_overground(V, u[i], v[i], x[i:(i+2)], y[i:(i+2)])

                while abs(end_velocity - start_velocity) > tolerance:
                    mid_velocity = (start_velocity + end_velocity) / 2
                    mid_drift_angle = self.find_drift_angle(mid_velocity, u[i], v[i], x[i:(i+2)], y[i:(i+2)])
                    mid_power = (self.HM(mid_velocity, depth[i], mid_drift_angle)[0] / 1000)
                    if mid_power > power:
                        end_velocity = mid_velocity
                    else:
                        start_velocity = mid_velocity
                V = (start_velocity + end_velocity) / 2
                velocity.append(V)
                drift_angle.append(self.find_drift_angle(V, u[i], v[i], x[i:(i+2)], y[i:(i+2)]))
                speed_overground.append(self.speed_overground(V, u[i], v[i], x[i:(i+2)], y[i:(i+2)]))
                time.append(t)

        else:
            for i in range(len(depth)-1):
                power = self.P
                start_velocity = 0
                end_velocity = 10
                tolerance = 10 ** -3

                if i > 0: 
                    t += ((x[i] - x[i-1]) * 1000) / (V + self.Prandle_Heaps(t, x[i], depth[i], self.T)[0])
                
                u, v = self.Prandle_Heaps(t, x[i], depth[i], self.T)[0:2]

                while abs(end_velocity - start_velocity) > tolerance:
                    mid_velocity = (start_velocity + end_velocity) / 2
                    mid_drift_angle = self.find_drift_angle(mid_velocity, u, v, x[i:(i+2)], y[i:(i+2)])
                    mid_power = (self.HM(mid_velocity, depth[i], mid_drift_angle)[0] / 1000)
                    if mid_power > power:
                        end_velocity = mid_velocity
                    else:
                        start_velocity = mid_velocity
                
                V = (start_velocity + end_velocity) / 2
                velocity.append(V)
                drift_angle.append(self.find_drift_angle(V, u, v, x[i:(i+2)], y[i:(i+2)]))
                speed_overground.append(V + self.Prandle_Heaps(t, x[i], depth[i], self.T)[0])
                time.append(t)
 
        return velocity, drift_angle, speed_overground, time

    def Prandle_Heaps(self, t, x, D, T):
        N = self.N
        
            #Coast Characteristics
            
        zeta = 0.2    
        h = D
        k = 0.0025
        f = 1.09083078 * 10**-4
        w = 1.4 * 10**-4
        g = 9.81
        drho_dx = - 2 * 10**-4
        rho = 1000
        U = 0.43
        s = (8 * k * U) / (3 * np.pi * N)
        z = np.linspace((D-T), D, 100)
        z_he = np.linspace(0, T, 100)

        u_p = []
        v_p = []
        u_h = []
        v_h = []

        for i in range(len(z)):

                #Heaps
            
            Dh = np.pi * ((2 * N)/ f)**0.5
            Z = z_he[i] + zeta
            H = h + zeta
            eta = Z / H
            a = np.pi * H / Dh
            a1 = a * (1 - eta)
            a2 = a * eta
            b = k * (H / N)
            C = a * (np.sinh(a) * np.cos(a) - np.cosh(a) * np.sin(a)) + b * np.cosh(a) * np.cos(a)
            E = a * (np.sinh(a) * np.cos(a) + np.cosh(a) * np.sin(a)) + b * np.sinh(a) * np.sin(a)
            L = b * np.cosh(a2) * np.cos(a2)
            M = b * np.sinh(a2) * np.sin(a2)
            P = C / (C**2 + E**2)
            Q = E / (C**2 + E**2)
            R = P * np.cosh(a) * np.cos(a) + Q * np.sinh(a) * np.sin(a)
            S = 1 - (R * b)
            Delta = (R - P - S) / S
            Lambda = 1 + b + b * Delta
            X = np.cosh(a1) * np.cos(a1) + (b / (2 * a)) * (np.sinh(a1) * np.cos(a1) + np.cosh(a1) * np.sin(a1)) - Lambda * np.cosh(a2) * np.cos(a2)
            Y = np.sinh(a1) * np.sin(a1) + (b / (2 * a)) * (np.cosh(a1) * np.sin(a1) - np.sinh(a1) * np.cos(a1)) - Lambda * np.sinh(a2) * np.sin(a2)
            
            u_he = (((g * H) / f) * (X * Q - Y * P) * (drho_dx) / rho) * np.exp(x / 50)
            v_he = (((g * H) / f) * (X * P + Y * Q + Delta + eta) * (drho_dx) / rho) * np.exp(x / 50)

            u_h.append(u_he)
            v_h.append(v_he)
        
                #Prandle
            
            a_plus = (1 + complex(0, 1)) * cmath.sqrt((f + w) / (2 * N))
            R_plus = np.cosh(a_plus * (z[i] - D)) - np.cosh(a_plus * D) - (a_plus / s) * np.sinh(a_plus * D) 
            R_plus_avg =  - np.cosh(a_plus * D) + (1 / (a_plus * D) - (a_plus / s)) * np.sinh(a_plus * D)

            a_min = (1 + complex(0, 1)) * cmath.sqrt((f - w) / (2 * N))
            R_min = np.cosh(a_min * (z[i] - D)) - np.cosh(a_min * D) - (a_min / s) * np.sinh(a_min * D) 
            R_min_avg =  - np.cosh(a_min * D) + (1 / (a_min * D) - (a_min / s)) * np.sinh(a_min * D)

            R_plus_plot = R_plus / R_plus_avg
            R_min_plot = R_min / R_min_avg 
            
            R_plus_amp = np.sqrt(R_plus_plot.real**2 + R_plus_plot.imag**2) * U
            R_min_amp = np.sqrt(R_min_plot.real**2 + R_min_plot.imag**2) * U

            R_plus_time = R_plus_amp * np.exp(-complex(0, 1) * w * t) 
            R_min_time = R_min_amp * np.exp(complex(0, 1) * w * t)
            R = R_plus_time + R_min_time
            
            u_pr = R.imag * np.exp(x / 50)
            v_pr = R.real * np.exp(x / 50)
            
            u_p.append(u_pr)
            v_p.append(v_pr)

        u_h = np.array(u_h)
        u_h = np.flip(u_h)
        v_h = np.array(v_h)
        v_h = np.flip(v_h)

        u_p = np.array(u_p)
        v_p = np.array(v_p)

        u = np.mean(u_p + u_h)
        v = np.mean(v_p + v_h)

        return u, v, u_p, u_h, v_p, v_h

    @staticmethod    
    def find_drift_angle(V, u, v, x, y):
        sailing_angle = np.arctan2((y[1] - y[0]), (x[1] - x[0]))
            
        current = np.array([u, v])
        SOW_vector = np.array([(V * np.cos(sailing_angle)), (V * np.sin(sailing_angle))])
        SOG_vector = current + SOW_vector
        
        correction_angle = np.arctan2(SOG_vector[1], SOG_vector[0])
        drift_angle = abs(sailing_angle - correction_angle) * (180 / np.pi)
        if drift_angle < 1e-10:
            drift_angle = 0

        return drift_angle

    @staticmethod
    def speed_overground(V, u, v, x, y):
        sailing_angle = np.arctan2((y[1] - y[0]), (x[1] - x[0]))

        current = np.array([u, v])
        SOW_vector = np.array([(V * np.cos(sailing_angle)), (V * np.sin(sailing_angle))])
        SOG_vector = current + SOW_vector

        SOG_magnitude = np.sqrt(SOG_vector[0]**2 + SOG_vector[1]**2)

        correction_angle = np.arctan2(SOG_vector[1], SOG_vector[0])
        drift_angle = abs(sailing_angle - correction_angle)

        SOG_magnitude = SOG_magnitude * np.cos(drift_angle)

        return SOG_magnitude