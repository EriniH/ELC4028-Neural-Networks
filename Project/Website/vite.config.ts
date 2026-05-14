import { defineConfig } from "@lovable.dev/vite-tanstack-config";

export default defineConfig({
  vite: {
    base: "/ELC4028-Neural-Networks/",
  },
  tanstackStart: {
    server: { entry: "server" },
  },
});