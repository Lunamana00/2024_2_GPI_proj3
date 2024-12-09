import pygame
import math
import numpy as np
import sys

def rotation_matrix_from_euler(rx, ry, rz):
    cx,sx=math.cos(rx),math.sin(rx)
    cy,sy=math.cos(ry),math.sin(ry)
    cz,sz=math.cos(rz),math.sin(rz)
    Rx=np.array([[1,0,0],
                 [0,cx,-sx],
                 [0,sx,cx]],dtype=float)
    Ry=np.array([[cy,0,sy],
                 [0,1,0],
                 [-sy,0,cy]],dtype=float)
    Rz=np.array([[cz,-sz,0],
                 [sz, cz,0],
                 [0,0,1]],dtype=float)
    return Rz@Ry@Rx

def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm<1e-10:
        return np.array([1,0,0,0],dtype=float)
    return q/norm

def quat_multiply(q1,q2):
    w1,x1,y1,z1=q1
    w2,x2,y2,z2=q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w,x,y,z],dtype=float)

def quat_from_axis_angle(axis,theta):
    axis=axis/np.linalg.norm(axis)
    w=math.cos(theta/2)
    s=math.sin(theta/2)
    return np.array([w, axis[0]*s, axis[1]*s, axis[2]*s],dtype=float)

def quat_to_matrix(q):
    w,x,y,z=q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),     1-2*(x*x+y*y)]
    ],dtype=float)

def axis_angle_from_angvel(angvel, dt):
    angle = np.linalg.norm(angvel)*dt
    if angle<1e-12:
        return np.array([1,0,0],dtype=float),0.0
    axis = angvel/np.linalg.norm(angvel)
    return axis, angle

def project_3d_to_2d(p, w, h, fov, vd):
    x,y,z=p
    denom=vd+z
    if abs(denom)<0.001:
        denom=0.001
    factor=fov/denom
    x2=x*factor + w/2
    y2=-y*factor + h/2
    return (int(x2),int(y2))

def camera_matrix_lookat(cam_pos, cam_target, up=np.array([0,1,0],dtype=float)):
    forward = cam_target - cam_pos
    forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,up)
    right=right/np.linalg.norm(right)
    up=np.cross(right,forward)
    R=np.eye(4,dtype=float)
    R[0,0:3]=right
    R[1,0:3]=up
    R[2,0:3]=-forward
    T=np.eye(4,dtype=float)
    T[0:3,3]=-cam_pos
    return R@T

def draw_ground_plane(surface, cam_mat, w,h,fov,vd):
    corners=[(-1000,-1000,0),
             (1000,-1000,0),
             (1000,1000,0),
             (-1000,1000,0)]
    proj=[]
    for c in corners:
        p_cam=cam_mat@np.append(c,1)
        proj.append(project_3d_to_2d(p_cam[:3],w,h,fov,vd))
    pygame.draw.polygon(surface,(50,50,50),proj,1)

def draw_xy_grid(surface, cam_mat,w,h,fov,vd, grid_range=500,step=100):
    line_color=(100,100,100)
    major=(150,150,150)
    for y in range(-grid_range,grid_range+1,step):
        p1=cam_mat@np.array([-grid_range,y,0,1],dtype=float)
        p2=cam_mat@np.array([grid_range,y,0,1],dtype=float)
        p1_2d=project_3d_to_2d(p1[:3],w,h,fov,vd)
        p2_2d=project_3d_to_2d(p2[:3],w,h,fov,vd)
        c=major if y==0 else line_color
        pygame.draw.line(surface,c,p1_2d,p2_2d,1)
    for x in range(-grid_range,grid_range+1,step):
        p1=cam_mat@np.array([x,-grid_range,0,1],dtype=float)
        p2=cam_mat@np.array([x,grid_range,0,1],dtype=float)
        p1_2d=project_3d_to_2d(p1[:3],w,h,fov,vd)
        p2_2d=project_3d_to_2d(p2[:3],w,h,fov,vd)
        c=major if x==0 else line_color
        pygame.draw.line(surface,c,p1_2d,p2_2d,1)

def draw_z_axis(surface, cam_mat,w,h,fov,vd):
    axis_length=200
    p1=cam_mat@np.array([0,0,-axis_length,1],dtype=float)
    p2=cam_mat@np.array([0,0, axis_length,1],dtype=float)
    p1_2d=project_3d_to_2d(p1[:3],w,h,fov,vd)
    p2_2d=project_3d_to_2d(p2[:3],w,h,fov,vd)
    pygame.draw.line(surface,(0,0,255),p1_2d,p2_2d,2)

def draw_views(screen, transformed, edges, cam_mats, w,h,fov,vd):
    sub_w=w//2
    sub_h=h//2
    def draw_view(cam,x,y):
        surf=pygame.Surface((sub_w,sub_h))
        surf.fill((0,0,0))
        draw_ground_plane(surf,cam,sub_w,sub_h,fov,vd)
        draw_xy_grid(surf,cam,sub_w,sub_h,fov,vd)
        draw_z_axis(surf,cam,sub_w,sub_h,fov,vd)
        for_view=[cam@np.append(v,1) for v in transformed]
        proj=[project_3d_to_2d(p[:3],sub_w,sub_h,fov,vd) for p in for_view]
        for e in edges:
            pygame.draw.line(surf,(255,255,255),proj[e[0]],proj[e[1]],1)
        screen.blit(surf,(x,y))
    draw_view(cam_mats[0],0,0)
    draw_view(cam_mats[1],sub_w,0)
    draw_view(cam_mats[2],0,sub_h)
    draw_view(cam_mats[3],sub_w,sub_h)

def get_vertices(shape):
    scale=100
    phi=(1+math.sqrt(5))/2
    if shape=='tetra':
        return np.array([
            [0,0,math.sqrt(2/3)],
            [0.5,0,-1/(2*math.sqrt(6))],
            [-0.5,0,-1/(2*math.sqrt(6))],
            [0,math.sqrt(3)/2,-1/(2*math.sqrt(6))]
        ],dtype=float)*scale
    elif shape=='cube':
        return np.array([
            [-0.5,-0.5,-0.5],
            [0.5,-0.5,-0.5],
            [0.5,0.5,-0.5],
            [-0.5,0.5,-0.5],
            [-0.5,-0.5,0.5],
            [0.5,-0.5,0.5],
            [0.5,0.5,0.5],
            [-0.5,0.5,0.5]
        ],dtype=float)*scale
    elif shape=='octa':
        return np.array([
            [0,0,1],
            [1,0,0],
            [0,1,0],
            [-1,0,0],
            [0,-1,0],
            [0,0,-1]
        ],dtype=float)*scale
    elif shape=='icosa':
        raw_verts = np.array([
            (0,   1,   phi),
            (0,  -1,   phi),
            (0,   1,  -phi),
            (0,  -1,  -phi),
            (1,   phi, 0),
            (-1,  phi, 0),
            (1,  -phi, 0),
            (-1,-phi,0),
            (phi, 0,   1),
            (-phi,0,   1),
            (phi, 0,  -1),
            (-phi,0,  -1)
        ],dtype=float)
        max_len=np.max(np.sqrt(np.sum(raw_verts**2,axis=1)))
        return (raw_verts/max_len)*scale
    else:
        # dodeca 제거 -> just return cube if else triggered
        return get_vertices('cube')

def get_edges(shape, verts):
    if shape=='tetra':
        return [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]
    elif shape=='cube':
        return [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    elif shape=='octa':
        return [(0,1),(0,2),(0,3),(0,4),(5,1),(5,2),(5,3),(5,4),(1,2),(2,3),(3,4),(4,1)]
    elif shape=='icosa':
        edges=[
            (0,1),(0,4),(0,5),(0,8),(0,9),
            (1,4),(1,6),(1,8),(1,9),
            (2,3),(2,5),(2,7),(2,10),(2,11),
            (3,6),(3,7),(3,10),(3,11),
            (4,5),(4,6),(4,8),(4,10),
            (5,7),(5,9),(5,11),
            (6,7),(6,8),(6,10),
            (7,9),(7,11),
            (8,9),(8,10),
            (9,11),
            (10,11)
        ]
        unique=set(tuple(sorted(e)) for e in edges)
        return list(unique)
    else:
        return get_edges('cube', verts)

def main():
    pygame.init()
    w,h=800,600
    screen=pygame.display.set_mode((w,h))
    pygame.display.set_caption("Fixed dtype issue, remove dodeca, no random torque if vertical")
    clock=pygame.time.Clock()
    font=pygame.font.SysFont(None,24)

    pygame.mouse.set_visible(True)

    fov=150
    vd=4.0

    shape_options=['tetra','cube','octa','icosa'] # dodeca 제거
    chosen_shape=None
    state='choose_shape'
    input_str=""

    gravity=9.8
    initial_velocity=[20.0,0.0,50.0]
    position=np.array([0.0,0.0,100.0],dtype=float)
    velocity=np.array([0.0,0.0,0.0],dtype=float)
    paused=False
    cam_pos=np.array([0.0,100.0,-300.0],dtype=float)
    cam_yaw=0.0
    cam_pitch=0.1
    cam_speed=100.0
    mouse_sens=0.0015

    rotation_q = np.array([1,0,0,0],dtype=float)
    angular_vel=np.array([0.0,0.0,0.0],dtype=float)

    shape_positions={
        'tetra':np.array([-800,0,500],dtype=float),
        'cube':np.array([-400,0,500],dtype=float),
        'octa':np.array([0,0,500],dtype=float),
        'icosa':np.array([400,0,500],dtype=float)
    }
    shape_bboxes={}

    instructions={
        'choose_shape':"Click on the shape you want to simulate",
        'input_vx':"Enter vx and press Enter:",
        'input_vy':"Enter vy and press Enter:",
        'input_vz':"Enter vz and press Enter:",
        'input_g':"Enter gravity g and press Enter:"
    }

    running=True
    while running:
        dt=clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False;break
            if state=='choose_shape':
                if event.type==pygame.MOUSEBUTTONDOWN:
                    mx,my=event.pos
                    for s in shape_options:
                        bbox=shape_bboxes.get(s,None)
                        if bbox:
                            (x1,y1,x2,y2)=bbox
                            if x1<=mx<=x2 and y1<=my<=y2:
                                chosen_shape=s
                                state='input_vx'
                                input_str=""
                                break
            elif state in ['input_vx','input_vy','input_vz','input_g']:
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_BACKSPACE:
                        input_str=input_str[:-1]
                    elif event.key==pygame.K_RETURN:
                        try:
                            val=float(input_str)
                            if state=='input_vx':
                                initial_velocity[0]=val
                                state='input_vy'
                            elif state=='input_vy':
                                initial_velocity[1]=val
                                state='input_vz'
                            elif state=='input_vz':
                                initial_velocity[2]=val
                                state='input_g'
                            elif state=='input_g':
                                gravity=val
                                velocity=np.array(initial_velocity,dtype=float)
                                paused=False
                                state='simulate'
                        except:
                            pass
                        input_str=""
                    else:
                        input_str+=event.unicode
            elif state=='simulate':
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_ESCAPE:
                        running=False;break
                    if event.key==pygame.K_SPACE:
                        paused=not paused
                    elif event.key==pygame.K_UP:
                        angular_vel[0]+=0.1
                    elif event.key==pygame.K_DOWN:
                        angular_vel[0]-=0.1
                    elif event.key==pygame.K_LEFT:
                        angular_vel[1]+=0.1
                    elif event.key==pygame.K_RIGHT:
                        angular_vel[1]-=0.1
                    elif event.key==pygame.K_q:
                        angular_vel[2]+=0.1
                    elif event.key==pygame.K_e:
                        angular_vel[2]-=0.1
                elif event.type==pygame.MOUSEMOTION:
                    rel=event.rel
                    cam_yaw-=rel[0]*mouse_sens
                    cam_pitch-=rel[1]*mouse_sens
                    cam_pitch=max(-math.pi/4,min(math.pi/4,cam_pitch))

        screen.fill((0,0,0))

        if state=='choose_shape':
            t=pygame.time.get_ticks()/1000.0
            rx,ry,rz=0,t,0
            R_euler=rotation_matrix_from_euler(rx,ry,rz)
            for s in shape_options:
                verts=get_vertices(s)
                eds=get_edges(s,verts)
                transformed=(verts@R_euler.T)+shape_positions[s]
                proj=[]
                for v in transformed:
                    proj.append(project_3d_to_2d(v,w,h,fov,vd))
                xs=[p[0] for p in proj]
                ys=[p[1] for p in proj]
                bbox=(min(xs),min(ys),max(xs),max(ys))
                shape_bboxes[s]=bbox
                for e in eds:
                    pygame.draw.line(screen,(255,255,255),proj[e[0]],proj[e[1]],1)
            info=font.render(instructions[state],True,(255,255,255))
            screen.blit(info,(20,20))

        elif state in ['input_vx','input_vy','input_vz','input_g']:
            if state=='input_vx':
                msg=f"Chosen shape: {chosen_shape}"
            elif state=='input_vy':
                msg=f"vx={initial_velocity[0]}"
            elif state=='input_vz':
                msg=f"vx={initial_velocity[0]}, vy={initial_velocity[1]}"
            elif state=='input_g':
                msg=f"vx={initial_velocity[0]}, vy={initial_velocity[1]}, vz={initial_velocity[2]}"
            m_surf=font.render(msg,True,(255,255,0))
            screen.blit(m_surf,(20,20))
            prompt=font.render(instructions[state],True,(255,255,255))
            screen.blit(prompt,(20,60))
            val_surf=font.render(input_str,True,(255,255,255))
            screen.blit(val_surf,(20,100))

        elif state=='simulate':
            keys=pygame.key.get_pressed()
            forward=np.array([math.sin(cam_yaw)*math.cos(cam_pitch),
                              -math.sin(cam_pitch),
                              math.cos(cam_yaw)*math.cos(cam_pitch)],dtype=float)
            right=np.array([math.cos(cam_yaw),0,-math.sin(cam_yaw)],dtype=float)
            up=np.cross(right,forward)

            if keys[pygame.K_w]:
                cam_pos+=forward*cam_speed*dt
            if keys[pygame.K_s]:
                cam_pos-=forward*cam_speed*dt
            if keys[pygame.K_a]:
                cam_pos-=right*cam_speed*dt
            if keys[pygame.K_d]:
                cam_pos+=right*cam_speed*dt
            if keys[pygame.K_SPACE]:
                cam_pos+=up*cam_speed*dt
            if keys[pygame.K_LSHIFT]:
                cam_pos-=up*cam_speed*dt

            if not paused:
                velocity[2]-=gravity*dt
                position=position+(velocity*dt)
                if position[2]<0:
                    position[2]=0
                    velocity[2]=-velocity[2]*0.8
                    # 토크 적용 개선:
                    # 수직으로 부딪히는 경우라면 수평 속도 거의 없음
                    vx_hor = np.sqrt(velocity[0]**2+velocity[1]**2)
                    if vx_hor>1e-3:
                        # 수평 성분이 있을 때만 토크 발생
                        rx=(np.random.rand()-0.5)*0.5*100
                        ry=(np.random.rand()-0.5)*0.5*100
                        contact=np.array([rx,ry,0],dtype=float)
                        force=np.array([0,0,abs(velocity[2])*2],dtype=float)
                        torque=np.cross(contact,force)
                        angular_vel+=torque*dt*0.1

                axis,angle=axis_angle_from_angvel(angular_vel, dt)
                dq=quat_from_axis_angle(axis,angle)
                rotation_q=quat_multiply(rotation_q,dq)
                rotation_q=quat_normalize(rotation_q)

            verts=get_vertices(chosen_shape)
            eds=get_edges(chosen_shape,verts)
            R=quat_to_matrix(rotation_q)
            transformed=(verts@R.T)+position

            cam_target=cam_pos+forward
            cam_mat_main=camera_matrix_lookat(cam_pos,cam_target,up)
            cam_pos_front=position+np.array([0,0,300],dtype=float)
            cam_t_front=position
            cam_mat_front=camera_matrix_lookat(cam_pos_front,cam_t_front,np.array([0,1,0],dtype=float))
            cam_pos_side=position+np.array([300,0,0],dtype=float)
            cam_t_side=position
            cam_mat_side=camera_matrix_lookat(cam_pos_side,cam_t_side,np.array([0,1,0],dtype=float))
            cam_pos_top=position+np.array([0,300,0],dtype=float)
            cam_t_top=position
            cam_mat_top=camera_matrix_lookat(cam_pos_top,cam_t_top,np.array([0,0,1],dtype=float))

            draw_views(screen,transformed,eds,[cam_mat_main,cam_mat_front,cam_mat_side,cam_mat_top],w,h,fov,vd)

            info1=font.render(f"Shape:{chosen_shape} | Pos:{position.round(2)} | Vel:{velocity.round(2)}",True,(255,255,0))
            screen.blit(info1,(20,20))
            info2=font.render(f"Gravity:{gravity:.2f} | Paused:{paused}",True,(255,255,0))
            screen.blit(info2,(20,40))
            info3=font.render(f"CamPos:{cam_pos.round(2)} Yaw:{cam_yaw:.2f} Pitch:{cam_pitch:.2f}",True,(255,255,0))
            screen.blit(info3,(20,60))
            info4=font.render("SPACE:pause, ESC:quit; Arrows/Q/E:rotate; W/S/A/D/SHIFT/SPACE:move cam; no dodeca, controlled torque",True,(255,255,255))
            screen.blit(info4,(20,80))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__=='__main__':
    main()
