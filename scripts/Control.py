from geometry_msgs.msg import Twist

class PID:
    def __init__(self,debug = False) -> None:
        '''
        input: goal point x,y (not angle value)
        output: Twist for cmd_vel
        '''

        self.debug = debug
        
        ## goal position
        self.goal_x = 0.
        self.goal_y = 0.

        ## gain settings
        self.p_x = 0.
        self.i_x = 0.
        self.d_x = 0.

        self.p_y = 0.
        self.i_y = 0.
        self.d_y = 0.

        ## error settings
        self.i_x_err = 0.
        self.i_y_err = 0.

        self.d_x_err = 0.
        self.d_y_err = 0.
        
        # output init
        self.twist = Twist()

        self.set_twist()


    def set_twist(self):
        
        # variables
        self.twist.linear.x = 0.
        self.twist.linear.y = 0.
        self.twist.angular.z = 0.

        #param
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0

    def set_input(self,px,ix,dx,py,iy,dy):
        ## x gains
        self.p_x = px
        self.i_x = ix
        self.d_x = dx
        ## y gains
        self.p_y = py
        self.i_y = iy
        self.d_y = dy

        print("PID gain initialized")
        print("--------------------------")
        self.get_input()

    def get_input(self):
        if self.debug:
            print("gains are:\n x: P = {:.2f} I = {:.2f} D = {:.2f} \n y: P = {:.2f} I = {:.2f} D = {:.2f}".format(self.p_x,self.i_x,self.d_x,self.p_y,self.i_y,self.d_y))
        
        return self.p_x,self.i_x,self.d_x,self.p_y,self.i_y,self.d_y
        

    def get_twist(self,goal_x,goal_y):
        ##foward
        # self.twist.linear.x = self.goal_x
        # self.twist.linear.y = self.goal_y

        if self.debug:
            str =  "***PID initialized***"
            print(str)

        p_err_x = goal_x
        p_err_y = goal_y

        self.i_x_err += goal_x
        self.i_y_err += goal_y

        d_error_x = p_err_x - self.d_x_err
        d_error_y = p_err_y - self.d_y_err

        self.d_x_err = d_error_x
        self.d_y_err = d_error_y

        self.twist.linear.x = self.p_x * p_err_x + self.i_x * self.i_x_err + self.d_x * self.d_x_err
        self.twist.linear.y = self.p_y * p_err_y + self.i_y * self.i_y_err + self.d_y * self.d_y_err

        if self.debug:
            print("publish twist")
        
        return self.twist


class MPC():
    def __init__(self) -> None:
        self.goal_x
        self.goal_y

        self.twist = Twist()

    def set_goal(self,x,y):
        self.goal_x = x
        self.goal_y = y
        pass


    