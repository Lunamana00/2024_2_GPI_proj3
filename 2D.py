import pygame
import math
from scipy.spatial import ConvexHull

pygame.init()

font = pygame.font.SysFont(None, 36)

# 화면 설정
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)

# 전역 변수
FPS = 60
points = []
polygon = None
friction = 0
restitution = 0.8  # 탄성계수
initial_force = [0.0, 0.0]
page = "shape_creation"
input_buffer = ""
input_target = None
dragging = False
obstacles = []
inside_polygon_drag = False
force_application_point = None
drag_start = None

def calculate_polygon_area(points):
    n = len(points)
    area = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def polygon_edges(points):
    return list(zip(points, points[1:] + points[:1]))

def point_in_polygon(point, polygon_points):
    x, y = point
    inside = False
    n = len(polygon_points)
    for i in range(n):
        x1, y1 = polygon_points[i]
        x2, y2 = polygon_points[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1)*(y - y1)/(y2 - y1) + x1):
            inside = not inside
    return inside

def line_intersect_circle(p1, p2, c, r):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = c

    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx*dx + dy*dy
    if a < 1e-8:
        # p1과 p2가 같은 점인 경우
        distance = math.hypot(x1 - cx, y1 - cy)
        if distance <= r:
            # 원 내부에 있으므로 충돌 발생
            return p1
        else:
            # 충돌 없음
            return None

    b = 2*(fx*dx + fy*dy)
    c_ = fx*fx + fy*fy - r*r

    discriminant = b*b - 4*a*c_
    if discriminant < 0:
        # 교차점 없음
        return None
    else:
        discriminant = math.sqrt(discriminant)
        t1 = (-b + discriminant) / (2*a)
        t2 = (-b - discriminant) / (2*a)

        # 선분 내의 유효한 t 값 찾기
        ts = [t for t in [t1, t2] if 0 <= t <= 1]
        if not ts:
            # 교차점이 선분 내에 없음
            return None

        # 가장 가까운 교차점 선택
        t_min = min(ts)
        ix = x1 + t_min*dx
        iy = y1 + t_min*dy
        return (ix, iy)

def reflect_velocity(velocity, normal):
    vx, vy = velocity
    nx, ny = normal
    length = math.sqrt(nx*nx + ny*ny)
    if length == 0:
        return velocity
    nx, ny = nx/length, ny/length
    dot = vx*nx + vy*ny
    vx_new = vx - 2*dot*nx
    vy_new = vy - 2*dot*ny
    vx_new *= restitution
    vy_new *= restitution
    return (vx_new, vy_new)

def calculate_polygon_inertia(points, area):
    n = len(points)
    inertia = 0.0
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        inertia += cross * (x0**2 + x0*x1 + x1**2 + y0**2 + y0*y1 + y1**2)
    return abs(inertia) / 12.0 / area

def is_convex(prev, current, next_):
    (px, py) = prev
    (cx, cy) = current
    (nx, ny) = next_
    cross = (cx - px) * (ny - py) - (cy - py) * (nx - px)
    return cross < 0

def point_in_triangle(pt, v1, v2, v3):
    """Check if a point is inside a triangle."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
               (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0

    return ((b1 == b2) and (b2 == b3))

def ear_clipping_triangulation(polygon_points):
    if len(polygon_points) < 3:
        return []

    vertices = polygon_points.copy()
    if calculate_polygon_area(vertices) < 0:
        vertices = vertices[::-1]

    triangles = []
    while len(vertices) > 3:
        ear_found = False
        n = len(vertices)
        for i in range(n):
            prev = vertices[i - 1]
            current = vertices[i]
            next_ = vertices[(i + 1) % n]

            if is_convex(prev, current, next_):
                ear = True
                for j in range(n):
                    if j in [(i - 1) % n, i, (i + 1) % n]:
                        continue
                    if point_in_triangle(vertices[j], prev, current, next_):
                        ear = False
                        break
                if ear:
                    triangles.append([prev, current, next_])
                    del vertices[i]
                    ear_found = True
                    break
        if not ear_found:
            break
    if len(vertices) == 3:
        triangles.append(vertices)
    return triangles

def triangle_circle_collision(triangle, circle_center, radius):
    for i in range(3):
        p1 = triangle[i]
        p2 = triangle[(i + 1) % 3]
        intersect = line_intersect_circle(p1, p2, circle_center, radius)
        if intersect:
            ix, iy = intersect
            nx = ix - circle_center[0]
            ny = iy - circle_center[1]
            length = math.hypot(nx, ny)
            if length == 0:
                continue 
            nx /= length
            ny /= length
            return (ix, iy), (nx, ny)
    if point_in_polygon(circle_center, triangle):
        cx, cy = circle_center
        tx, ty = triangle[0]
        nx = cx - tx
        ny = cy - ty
        length = math.hypot(nx, ny)
        if length == 0:
            return None
        nx /= length
        ny /= length
        return (cx, cy), (nx, ny)
    return None

class Polygon:
    def __init__(self, points, velocity, angular_velocity, friction):
        self.original_points = points.copy()
        self.points = points
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.friction = friction
        self.center = self.calculate_center()
        self.area = calculate_polygon_area(points)
        self.inertia = calculate_polygon_inertia(points, self.area)
        self.triangles = ear_clipping_triangulation(self.points)

    def calculate_center(self):
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return (sum(x_coords) / len(self.points), sum(y_coords) / len(self.points))

    def move(self):
        self.velocity = (self.velocity[0] * self.friction, self.velocity[1] * self.friction)
        self.points = [(x + self.velocity[0], y + self.velocity[1]) for x, y in self.points]
        self.center = self.calculate_center()
        self.triangles = ear_clipping_triangulation(self.points)

    def rotate(self):
        angle = self.angular_velocity
        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)
        cx, cy = self.center
        self.points = [
            (
                cos_theta * (x - cx) - sin_theta * (y - cy) + cx,
                sin_theta * (x - cx) + cos_theta * (y - cy) + cy,
            )
            for x, y in self.points
        ]
        self.triangles = ear_clipping_triangulation(self.points)

    def handle_collision(self, obstacles):
        for (x, y) in self.points:
            if x <= 0:
                normal = (1, 0)
                self.velocity = reflect_velocity(self.velocity, normal)
                self.angular_velocity *= -0.5
                self.points = [(px + 1, py) for (px, py) in self.points]
                self.center = self.calculate_center()
                self.triangles = ear_clipping_triangulation(self.points)
                break
            if x >= WIDTH:
                normal = (-1, 0)
                self.velocity = reflect_velocity(self.velocity, normal)
                self.angular_velocity *= -0.5
                self.points = [(px - 1, py) for (px, py) in self.points]
                self.center = self.calculate_center()
                self.triangles = ear_clipping_triangulation(self.points)
                break
            if y <= 0:
                normal = (0, 1)
                self.velocity = reflect_velocity(self.velocity, normal)
                self.angular_velocity *= -0.5
                self.points = [(px, py + 1) for (px, py) in self.points]
                self.center = self.calculate_center()
                self.triangles = ear_clipping_triangulation(self.points)
                break
            if y >= HEIGHT:
                normal = (0, -1)
                self.velocity = reflect_velocity(self.velocity, normal)
                self.angular_velocity *= -0.5
                self.points = [(px, py - 1) for (px, py) in self.points]
                self.center = self.calculate_center()
                self.triangles = ear_clipping_triangulation(self.points)
                break

        # 원형 장애물과의 충돌 감지 및 응답
        for obstacle in obstacles:
            ox, oy, r = obstacle['x'], obstacle['y'], obstacle['r']
            for triangle in self.triangles:
                collision_info = triangle_circle_collision(triangle, (ox, oy), r)
                if collision_info:
                    ip, normal = collision_info
                    self.resolve_collision(normal, obstacle)
                    break 

    def resolve_collision(self, normal, obstacle):
        relative_velocity = (
            self.velocity[0] - obstacle['vx'],
            self.velocity[1] - obstacle['vy']
        )

        rel_vel_dot_n = relative_velocity[0] * normal[0] + relative_velocity[1] * normal[1]

        if rel_vel_dot_n > 0:
            return  # 서로 멀어지고 있으므로 충돌 처리하지 않음

        # 임펄스 크기 계산
        impulse_mag = -(1 + restitution) * rel_vel_dot_n
        impulse_mag /= (1 / 1.0 + 1 / 1.0)  # 질량 = 1 가정

        # 임펄스 벡터 계산
        Jx = impulse_mag * normal[0]
        Jy = impulse_mag * normal[1]

        # 폴리곤의 속도 업데이트
        self.velocity = (
            self.velocity[0] + Jx / 1.0,
            self.velocity[1] + Jy / 1.0
        )

        # 장애물의 속도 업데이트 (반대 방향으로)
        obstacle['vx'] -= Jx / 1.0
        obstacle['vy'] -= Jy / 1.0

        # 충돌 지점에서 각속도 업데이트
        # 충돌 지점과 다각형 중심 간의 벡터를 계산
        collision_point = (
            obstacle['x'] - normal[0] * obstacle['r'],
            obstacle['y'] - normal[1] * obstacle['r']
        )
        cx, cy = self.center
        rx = collision_point[0] - cx
        ry = collision_point[1] - cy

        # 충격량(momentum)에 따라 각속도를 업데이트
        torque = rx * (Jy) - ry * (Jx)
        self.angular_velocity += torque / self.inertia

        # 충돌 후 각속도 약간 감소
        self.angular_velocity *= restitution

    def draw(self, screen):
        # 폴리곤 그리기
        pygame.draw.polygon(screen, RED, self.points, 0)
        for triangle in self.triangles:
            pygame.draw.polygon(screen, GREEN, triangle, 1)

    def apply_force_at_point(self, point, force):
        cx, cy = self.center
        px, py = point
        rx = px - cx
        ry = py - cy

        fx, fy = force
        self.velocity = (self.velocity[0] + fx, self.velocity[1] + fy)

        torque = rx * fy - ry * fx 
        self.angular_velocity += 10*torque / self.inertia

def handle_obstacle_collisions(obstacles, restitution):
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            obs1 = obstacles[i]
            obs2 = obstacles[j]
            dx = obs2['x'] - obs1['x']
            dy = obs2['y'] - obs1['y']
            distance = math.hypot(dx, dy)
            min_dist = obs1['r'] + obs2['r']
            if distance < min_dist and distance != 0:
                # 노말 벡터 계산
                nx = dx / distance
                ny = dy / distance

                # 상대 속도 계산
                dvx = obs1['vx'] - obs2['vx']
                dvy = obs1['vy'] - obs2['vy']
                rel_vel = dvx * nx + dvy * ny

                if rel_vel > 0:
                    continue  

                # 임펄스 크기 계산
                impulse = -(1 + restitution) * rel_vel
                impulse /= (1 / 1.0 + 1 / 1.0)  # 질량 = 1 가정

                # 임펄스 벡터 계산
                Jx = impulse * nx
                Jy = impulse * ny

                # 장애물의 속도 업데이트
                obs1['vx'] += Jx / 1.0
                obs1['vy'] += Jy / 1.0
                obs2['vx'] -= Jx / 1.0
                obs2['vy'] -= Jy / 1.0

                # 위치 보정 (붙는 현상 방지)
                overlap = min_dist - distance
                correction = overlap / 2
                obs1['x'] -= correction * nx
                obs1['y'] -= correction * ny
                obs2['x'] += correction * nx
                obs2['y'] += correction * ny

    return obstacles

clock = pygame.time.Clock()
running = True

while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

        if page == "shape_creation":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                points.append(event.pos)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if len(points) > 2:
                    page = "settings"

        elif page == "settings":
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and input_target is None:
                    page = "simulation"
                    polygon = Polygon(points, velocity=initial_force, angular_velocity=0.05, friction=1-friction)
                elif event.key == pygame.K_BACKSPACE:
                    input_buffer = input_buffer[:-1]
                elif event.key == pygame.K_RETURN and input_target:
                    try:
                        value = float(input_buffer)
                        if input_target == "friction":
                            friction = value
                        elif input_target == "force_x":
                            initial_force[0] = value
                        elif input_target == "force_y":
                            initial_force[1] = value
                        elif input_target == "restitution":
                            restitution = value
                    except ValueError:
                        pass
                    input_buffer = ""
                    input_target = None
                else:
                    input_buffer += event.unicode

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if 50 <= my <= 80:
                    input_target = "friction"
                    input_buffer = ""
                elif 100 <= my <= 130:
                    input_target = "force_x"
                    input_buffer = ""
                elif 150 <= my <= 180:
                    input_target = "force_y"
                    input_buffer = ""
                elif 200 <= my <= 230:
                    input_target = "restitution"
                    input_buffer = ""

        elif page == "simulation":
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    # 오른쪽 클릭: 원형 장애물 생성, 초기 속도 0
                    mouse_pos = pygame.mouse.get_pos()
                    obstacles.append({'x': mouse_pos[0], 'y': mouse_pos[1], 'r': 15, 'vx': 0.0, 'vy': 0.0})
                elif event.button == 1:
                    click_pos = pygame.mouse.get_pos()
                    if polygon and point_in_polygon(click_pos, polygon.points):
                        inside_polygon_drag = True
                        force_application_point = click_pos
                        drag_start = click_pos
                        dragging = True
                    else:
                        inside_polygon_drag = False
                        drag_start = click_pos
                        dragging = True

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                drag_end = pygame.mouse.get_pos()
                dx = drag_end[0] - drag_start[0]
                dy = drag_end[1] - drag_start[1]

                if polygon:
                    if inside_polygon_drag:
                        force = (dx / 10.0, dy / 10.0)
                        polygon.apply_force_at_point(force_application_point, force)
                    else:
                        polygon.velocity = (dx / 10, dy / 10)

                dragging = False
                inside_polygon_drag = False
                force_application_point = None

    # 페이지별 화면 처리
    if page == "shape_creation":
        for point in points:
            pygame.draw.circle(screen, BLUE, point, 5)
        if len(points) > 1:
            pygame.draw.lines(screen, BLACK, False, points, 2)
        msg = "Click to create points. Press ENTER when done."
        screen.blit(font.render(msg, True, BLACK), (10, 10))

    elif page == "settings":
        screen.blit(font.render(f"Friction: {friction:.2f}", True, BLACK), (50, 50))
        screen.blit(font.render(f"Force X: {initial_force[0]:.2f}", True, BLACK), (50, 100))
        screen.blit(font.render(f"Force Y: {initial_force[1]:.2f}", True, BLACK), (50, 150))
        screen.blit(font.render(f"Restitution: {restitution:.2f}", True, BLACK), (50, 200))

        if input_target:
            screen.blit(font.render(f"Enter {input_target}: {input_buffer}", True, GREEN), (50, 250))
        else:
            msg = "Click a value to edit. Press ENTER to start simulation."
            screen.blit(font.render(msg, True, BLACK), (50, 250))

    elif page == "simulation":
        # 폴리곤 업데이트 및 그리기
        if polygon:
            polygon.move()
            polygon.rotate()
            polygon.handle_collision(obstacles)
            polygon.draw(screen)

        # 장애물 업데이트 (움직이게)
        for obs in obstacles:
            # 마찰 적용
            obs['vx'] *= 1-friction
            obs['vy'] *= 1-friction
            # 위치 업데이트
            obs['x'] += obs['vx']
            obs['y'] += obs['vy']

            # 화면 경계 처리 (간단 반사)
            if obs['x'] - obs['r'] < 0:
                obs['x'] = obs['r']
                obs['vx'] = -obs['vx'] * restitution
            if obs['x'] + obs['r'] > WIDTH:
                obs['x'] = WIDTH - obs['r']
                obs['vx'] = -obs['vx'] * restitution
            if obs['y'] - obs['r'] < 0:
                obs['y'] = obs['r']
                obs['vy'] = -obs['vy'] * restitution
            if obs['y'] + obs['r'] > HEIGHT:
                obs['y'] = HEIGHT - obs['r']
                obs['vy'] = -obs['vy'] * restitution

        # 장애물 간의 충돌 처리
        obstacles = handle_obstacle_collisions(obstacles, restitution)

        # 장애물 그리기
        for obs in obstacles:
            pygame.draw.circle(screen, GRAY, (int(obs['x']), int(obs['y'])), obs['r'])  # 내부 채우기
            pygame.draw.circle(screen, BLACK, (int(obs['x']), int(obs['y'])), obs['r'], 2)  # 테두리

        # 드래그 표시
        if dragging and drag_start:
            current_pos = pygame.mouse.get_pos()
            pygame.draw.line(screen, GREEN, drag_start, current_pos, 2)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
