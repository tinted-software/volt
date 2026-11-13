#include "EGL/egl.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DEMO_HEADLESS
#include <SDL3/SDL.h>
#endif

#ifdef _WIN32
#include <windows.h>
#define sleep_ms(ms) Sleep(ms)
#else
#include <unistd.h>
#define sleep_ms(ms) usleep((ms) * 1000)
#endif

static GLuint compile_cube_list(void) {
  GLuint list = glGenLists(1);
  glNewList(list, GL_COMPILE);
  glBegin(GL_QUADS);

  // Front face
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
  glVertex3f(-1.0f, -1.0f, 1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(1.0f, 0.2f, 0.2f, 1.0f);
  glVertex3f(1.0f, -1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(1.0f, 0.5f, 0.5f, 1.0f);
  glVertex3f(-1.0f, 1.0f, 1.0f);

  // Back face
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(0.2f, 1.0f, 0.2f, 1.0f);
  glVertex3f(-1.0f, -1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(0.5f, 1.0f, 0.5f, 1.0f);
  glVertex3f(-1.0f, 1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(0.2f, 0.2f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, -1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(0.5f, 0.5f, 0.5f, 1.0f);
  glVertex3f(1.0f, -1.0f, -1.0f);

  // Top face
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(0.2f, 0.2f, 1.0f, 1.0f);
  glVertex3f(-1.0f, 1.0f, -1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(0.2f, 0.2f, 1.0f, 1.0f);
  glVertex3f(-1.0f, 1.0f, 1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, -1.0f);

  // Bottom face
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(1.0f, 1.0f, 0.2f, 1.0f);
  glVertex3f(-1.0f, -1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(1.0f, 1.0f, 0.2f, 1.0f);
  glVertex3f(-1.0f, -1.0f, 1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(1.0f, 1.0f, 0.5f, 1.0f);
  glVertex3f(1.0f, -1.0f, 1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(1.0f, 1.0f, 0.5f, 1.0f);
  glVertex3f(1.0f, -1.0f, 1.0f);

  // Right face
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(1.0f, 0.2f, 1.0f, 1.0f);
  glVertex3f(1.0f, -1.0f, -1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(1.0f, 0.5f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
  glVertex3f(1.0f, 1.0f, 1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(0.2f, 0.2f, 1.0f, 1.0f);
  glVertex3f(1.0f, -1.0f, 1.0f);

  // Left face
  glTexCoord2f(0.0f, 0.0f);
  glColor4f(0.2f, 1.0f, 1.0f, 1.0f);
  glVertex3f(-1.0f, -1.0f, -1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glColor4f(0.2f, 1.0f, 1.0f, 1.0f);
  glVertex3f(-1.0f, -1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
  glVertex3f(-1.0f, 1.0f, 1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glColor4f(0.5f, 0.5f, 1.0f, 1.0f);
  glVertex3f(-1.0f, 1.0f, -1.0f);

  glEnd();
  glEndList();
  return list;
}

int main(int argc, char **argv) {
  bool headless_mode = false;
  int max_frames = 0; // 0 = run until window close

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--headless") == 0 || strcmp(argv[i], "-h") == 0) {
      headless_mode = true;
    } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
      max_frames = atoi(argv[++i]);
    }
  }

#ifndef DEMO_HEADLESS
  if (!headless_mode && !SDL_Init(SDL_INIT_VIDEO)) {
    fprintf(stderr, "SDL_Init failed: %s — falling back to headless\n",
            SDL_GetError());
    headless_mode = true;
  }
#endif

  printf("========================================\n");
  printf(" angle_wgpu C Demo (SDL3)\n");
  printf(" Mode: %s\n",
         headless_mode ? "Headless Offscreen" : "Windowed (SDL3)");
  printf("========================================\n\n");

  // 1. Initialize EGL
  printf("[1] Initializing EGL display...\n");
  EGLDisplay dpy = eglGetDisplay(NULL);
  if (!dpy) {
    fprintf(stderr, "Failed to get EGL display\n");
    return 1;
  }

  EGLint major = 0, minor = 0;
  if (!eglInitialize(dpy, &major, &minor)) {
    fprintf(stderr, "Failed to initialize EGL: 0x%x\n", eglGetError());
    return 1;
  }
  printf("    EGL version: %d.%d\n", major, minor);

  // 2. Choose Config
  printf("[2] Choosing EGL config...\n");
  EGLConfig config = NULL;
  EGLint num_configs = 0;
  EGLint config_attribs[] = {EGL_NONE};
  if (!eglChooseConfig(dpy, config_attribs, &config, 1, &num_configs) ||
      num_configs < 1) {
    fprintf(stderr, "Failed to choose EGL config\n");
    return 1;
  }

  // 3. Create Surface
  EGLSurface surf = NULL;
  uint32_t width = 800;
  uint32_t height = 600;

#ifndef DEMO_HEADLESS
  SDL_Window *window = NULL;
#endif

#ifndef DEMO_HEADLESS
  if (!headless_mode) {
    printf("[3] Opening SDL3 window (800x600)...\n");
    window = SDL_CreateWindow("angle_wgpu Demo", (int)width, (int)height,
                              SDL_WINDOW_RESIZABLE);
    if (!window) {
      fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
      return 1;
    }

    // Extract the platform-native window handles from SDL's window
    // properties and hand them straight to the EGL layer.
    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    AngleWgpuNativeWindow native;
    memset(&native, 0, sizeof(native));
#if defined(_WIN32)
    native.kind = ANGLE_WGPU_NATIVE_WIN32;
    native.display = SDL_GetPointerProperty(
        props, SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, NULL);
    native.window = (uint64_t)(uintptr_t)SDL_GetPointerProperty(
        props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    native.width = width;
    native.height = height;
#elif defined(__APPLE__)
    fprintf(stderr,
            "Native window export not wired up for this platform yet\n");
    SDL_DestroyWindow(window);
    return 1;
#else
    const char *vid_drv = SDL_GetCurrentVideoDriver();
    if (vid_drv && strcmp(vid_drv, "x11") == 0) {
      native.kind = ANGLE_WGPU_NATIVE_X11;
      native.display = SDL_GetPointerProperty(
          props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
      native.window =
          SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
      native.screen = (int32_t)SDL_GetNumberProperty(
          props, SDL_PROP_WINDOW_X11_SCREEN_NUMBER, 0);
      native.width = width;
      native.height = height;
    } else if (vid_drv && strcmp(vid_drv, "wayland") == 0) {
      native.kind = ANGLE_WGPU_NATIVE_WAYLAND;
      native.display = SDL_GetPointerProperty(
          props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);
      native.window = (uint64_t)(uintptr_t)SDL_GetPointerProperty(
          props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
      native.width = width;
      native.height = height;
    } else {
      fprintf(stderr,
              "Unsupported video driver '%s' for native surface export\n",
              vid_drv ? vid_drv : "(null)");
      SDL_DestroyWindow(window);
      return 1;
    }
#endif

    surf = angle_wgpu_create_native_window_surface(dpy, config, &native);
    if (!surf) {
      fprintf(stderr,
              "Failed to create native window EGL surface (eglGetError=0x%x)\n",
              eglGetError());
      SDL_DestroyWindow(window);
      return 1;
    }
  }
#endif
  if (!surf) {
    printf("[3] Creating EGL PBuffer Surface (800x600)...\n");
    EGLint pbuffer_attribs[] = {EGL_WIDTH, (EGLint)width, EGL_HEIGHT,
                                (EGLint)height, EGL_NONE};
    surf = eglCreatePbufferSurface(dpy, config, pbuffer_attribs);
    if (!surf) {
      fprintf(stderr, "Fatal: Could not create any EGL surface\n");
      return 1;
    }
  }

  // 4. Create Context and Make Current
  printf("[4] Creating EGL Context and binding to surface...\n");
  EGLContext ctx = eglCreateContext(dpy, config, NULL, NULL);
  if (!ctx) {
    fprintf(stderr, "Failed to create EGL context\n");
    return 1;
  }

  if (!eglMakeCurrent(dpy, surf, surf, ctx)) {
    fprintf(stderr, "Failed to make EGL context current\n");
    return 1;
  }

  // 5. Query GL Info
  const GLubyte *vendor = glGetString(GL_VENDOR);
  const GLubyte *renderer = glGetString(GL_RENDERER);
  const GLubyte *version = glGetString(GL_VERSION);
  printf("    GL Vendor:   %s\n", vendor ? (const char *)vendor : "null");
  printf("    GL Renderer: %s\n", renderer ? (const char *)renderer : "null");
  printf("    GL Version:  %s\n\n", version ? (const char *)version : "null");

  // 6. Texture (checkerboard pattern)
  printf("[5] Uploading texture...\n");
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);

#define PAT_W 8
#define PAT_H 8
  uint32_t pattern_pixels[PAT_W * PAT_H];
  for (int y = 0; y < PAT_H; ++y) {
    for (int x = 0; x < PAT_W; ++x) {
      bool check = ((x + y) % 2) == 0;
      pattern_pixels[y * PAT_W + x] = check ? 0xFF30D5C8 : 0xFFFF4500;
    }
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, PAT_W, PAT_H, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, pattern_pixels);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glEnable(GL_TEXTURE_2D);

  // 7. Compile 3D Cube Display List
  printf("[6] Compiling 3D Rotating Cube Display List...\n");
  GLuint cube_list = compile_cube_list();

  // 8. Render Loop
  printf("[7] Starting render loop...\n");
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  float rot_x = 0.0f, rot_y = 0.0f, rot_z = 0.0f;
  int frame_idx = 0;

  while (true) {
#ifndef DEMO_HEADLESS
    if (window) {
      SDL_Event ev;
      while (SDL_PollEvent(&ev)) {
        if (ev.type == SDL_EVENT_QUIT ||
            ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
          goto done;
        }
      }
      int win_w = 0, win_h = 0;
      SDL_GetWindowSize(window, &win_w, &win_h);
      if (win_w > 0 && win_h > 0 &&
          ((uint32_t)win_w != width || (uint32_t)win_h != height)) {
        width = (uint32_t)win_w;
        height = (uint32_t)win_h;
        // Resize the swapchain to match the window.
        angle_wgpu_resize_surface(surf, width, height);
      }
    }
#endif

    glViewport(0, 0, (GLsizei)width, (GLsizei)height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)width / (float)height;
    glOrthof(-3.0f * aspect, 3.0f * aspect, -3.0f, 3.0f, -20.0f, 20.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -5.0f);
    glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
    glRotatef(rot_y, 0.0f, 1.0f, 0.0f);
    glRotatef(rot_z, 0.0f, 0.0f, 1.0f);

    glClearColor(0.12f, 0.15f, 0.22f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glCallList(cube_list);

    eglSwapBuffers(dpy, surf);

    rot_x += 0.8f;
    rot_y += 1.2f;
    rot_z += 0.5f;
    frame_idx++;

    if (max_frames > 0 && frame_idx >= max_frames) {
      printf("Reached target frame limit (%d frames).\n", max_frames);
      break;
    }

    sleep_ms(16);
  }

done:
  // 9. Teardown
  printf("\n[8] Cleaning up resources...\n");
  glDeleteLists(cube_list, 1);
  glDeleteTextures(1, &tex);

  eglMakeCurrent(dpy, NULL, NULL, NULL);
  eglDestroyContext(dpy, ctx);
  eglDestroySurface(dpy, surf);
  eglTerminate(dpy);

#ifndef DEMO_HEADLESS
  if (window) {
    SDL_DestroyWindow(window);
    SDL_Quit();
  }
#endif

  printf("Demo finished successfully.\n");
  return 0;
}
