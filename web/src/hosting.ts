/** Static public builds play recordings; live capture belongs to the local controller. */
export const HOSTED_DEMO = import.meta.env?.VITE_HOSTED_DEMO === "true"
export const LOCAL_CAPTURE_MESSAGE = "Live capture runs in the local app and controller. Clone the repository to connect your own GPU host."
