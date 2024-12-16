import type { PostFilter } from "./utils/posts";

export interface SiteConfig {
  title: string;
  slogan: string;
  description?: string;
  site: string,
  social: {
    github?: string;
    bilibili?: string;
    weibo?: string;
    email?: string;
  };
  homepage: PostFilter;
  googleAnalysis?: string;
}

export const siteConfig: SiteConfig = {
  site: "https://yangzhe.us/", // your site url
  title: "Yangzhe's blog",
  slogan: "他人之智 贯通之极",
  description: "",
  social: {
    github: "https://github.com/yangzhe0", // leave empty if you don't want to show the github
    bilibili: "https://space.bilibili.com/290208229", // leave empty if you don't want to show the linkedin
    weibo: "https://www.weibo.com/7527059128",
    email: "zhe9052@gmail.com",
  },
  homepage: {
    maxPosts: 5,
    tags: ['Work'],
    excludeTags: [],
  },
  googleAnalysis: "", // your google analysis id
};
