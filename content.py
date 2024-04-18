Void CV_filter::Initialize_Filter_state_covarience(float x,float y,float z,float vx,float vy,float vz,float time)
{
    int i,j,k,l;
    Sf.put_Element(x,0,0)
    Sf.put_Element(y,1,0)
    Sf.put_Element(z,2,0)
    Sf.put_Element(vx,3,0)
    Sf.put_Element(vy,4,0)
    Sf.put_Element(vz,5,0)
    this->Filtered_Time=time;
    for(i=0;i<6;i++)
    {
        for(j=0;j<6:j++)
        {
            k=i%3;
            l=j%3;
            pf.put_Element(R.Get_Element(K,l),i,j);
        }
    }    
}

Void CV_filter::predict_state_covarience(float_det)
{
Float T_3,T_2
Phi.put_Element(1,0,0);
Phi.put_Element(delt,0,3)
Phi.put_Element(1,1,1)
Phi.put_Element(delt,1,4)
Phi.put_Element(1,2,2)
Phi.put_Element(delt,2,5)
Phi.put_Element(1,3,3)
Phi.put_Element(1,4,4)
Phi.put_Element(1,5,5)
Sp=Phi*sf;
predicted_Time=Filtered_time+delt;
T_3=(delt*delt*delt)/3.0;
T_2=(delt*delt)/2.0;
Q.put_Element(T_3,0,0);
Q.put_Element(T_3,1,1);
Q.put_Element(T_3,2,2);
Q.put_Element(T_2,0,3);
Q.put_Element(T_2,1,4);
Q.put_Element(T_2,2,5);
Q.put_Element(T_2,3,0);
Q.put_Element(T_2,4,1);
Q.put_Element(T_2,5,2);
Q.put_Element(delt,3,3);
Q.put_Element(delt,4,4);
Q.put_Element(delt,5,5);
Q=Q*plant_noise;
Pp=Phi*pf*(Phi.matrix_transpose())+Q
}

Void CV_filter::Filter_state_covarience()
{
    Prev_Sf=Sf;
    Prev_Filtered.Time=Filtered_time;
    S=R+H*Pp*(H.matrix_Transpose());
    K=(Pp*(H.matrix_Transpose()))*(S.matrix_inverse());
    Inn=(Z-H*Sp);
    Sf=Sp+K*Inn;
    pf=(Inn-K*H)*Pp;
    Filtered_Time=Meas_Time;
}

sig_e_sqr=

#initializing R Matrix

rpt.R.d_00=sig_r*sig_r*cos(e)*cos(e)*sin(a)*sin(a)+r*r*cos(e)*cos(e)*cos(a)*cos(a)+sig_a*sig_a+r*r*sin(e)*sin(e)*sin(a)*sin(a)*sig_e_sqr;

rpt.R.d_11=sig_r*sig_r*cos(e)*cos(e)*cos(a)*cos(a)+r*r*cos(e)*cos(e)*sin(a)*sin(a)+sig_a*sig_a+r*r*sin(e)*sin(e)*cos(a)*cos(a)*sig_e_sqr;

rpt.R.d_22=sig_r*sig_r*sin(e)*sin(e)+r*r*cos(e)*cos(e)*sig_e_sqr;












