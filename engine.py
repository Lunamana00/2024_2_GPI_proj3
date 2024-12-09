import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# -----------------------------
# 정다면체 데이터 정의
# -----------------------------
# 정사면체
tetrahedron_vertices = [
    [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
]
tetrahedron_faces = [
    (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
]

# 정육면체
cube_vertices = [
    [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
    [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
]
cube_faces = [
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)
]

# 정팔면체
octahedron_vertices = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
]
octahedron_faces = [
    (0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
    (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)
]

# 정이십면체
phi = (1 + np.sqrt(5)) / 2  # 황금비
icosahedron_vertices = [
    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
]
icosahedron_faces = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
]

# -----------------------------
# Polyhedron 클래스 정의
# -----------------------------
class Polyhedron:
    def __init__(self, vertices, faces, mass=1.0):
        self.original_vertices = np.array(vertices)
        self.faces = faces
        self.mass = mass
        self.position = np.array([0.0, 0.0, -5.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.force = np.array([0.0, 0.0, 0.0])

    def apply_force(self, force):
        self.force += np.array(force)

    def update(self, dt):
        # F = m * a => a = F / m
        acceleration = self.force / self.mass
        # Update velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        # Reset force after each update
        self.force = np.array([0.0, 0.0, 0.0])

    def get_transformed_vertices(self):
        # Apply translation to vertices
        transformed_vertices = self.original_vertices + self.position
        return transformed_vertices

# -----------------------------
# 도형 렌더링 함수
# -----------------------------
def draw_shape(polyhedron, wireframe=False):
    vertices = polyhedron.get_transformed_vertices()
    faces = polyhedron.faces

    # 면 렌더링
    glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 면을 채워서 렌더링
    glBegin(GL_TRIANGLES if len(faces[0]) == 3 else GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    if wireframe:
        # 테두리 렌더링
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # 테두리 모드로 전환
        glColor3f(0, 0, 0)  # 검은색 테두리
        glBegin(GL_LINES)
        for face in faces:
            for i in range(len(face)):
                glVertex3fv(vertices[face[i]])
                glVertex3fv(vertices[face[(i + 1) % len(face)]])
        glEnd()

# -----------------------------
# 메인 함수
# -----------------------------
def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Polyhedron Simulation")

    # OpenGL 설정
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -10)  # 카메라 위치 조정

    # 다면체 데이터
    shapes = {
        '1': (tetrahedron_vertices, tetrahedron_faces, "Tetrahedron"),
        '2': (cube_vertices, cube_faces, "Cube"),
        '3': (octahedron_vertices, octahedron_faces, "Octahedron"),
        '4': (icosahedron_vertices, icosahedron_faces, "Icosahedron")
    }

    current_shape_key = '1'
    vertices, faces, shape_name = shapes[current_shape_key]
    polyhedron = Polyhedron(vertices, faces)

    clock = pygame.time.Clock()
    is_running = True
    rotation_x, rotation_y = 0, 0

    while is_running:
        dt = clock.tick(60) / 1000.0  # 시간 간격 계산

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.KEYDOWN:
                # 다면체 선택
                if event.unicode in shapes:
                    current_shape_key = event.unicode
                    vertices, faces, shape_name = shapes[current_shape_key]
                    polyhedron = Polyhedron(vertices, faces, polyhedron.mass)

        # 키보드 입력 처리
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            rotation_y -= 1
        if keys[pygame.K_RIGHT]:
            rotation_y += 1
        if keys[pygame.K_UP]:
            rotation_x -= 1
        if keys[pygame.K_DOWN]:
            rotation_x += 1

        # 물리 시뮬레이션 업데이트
        polyhedron.update(dt)

        # 화면 갱신
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()

        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)

        draw_shape(polyhedron, wireframe=True)

        glPopMatrix()
        pygame.display.set_caption(f"Polyhedron Simulation - {shape_name}")
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
