---
title: Remote Streaming
description: Stream the CyberEther interface to a browser on another device, hosted entirely on your own machine.
order: 243
category: Usage
parent: platforms
---

CyberEther Remote streams the full interactive interface of a running CyberEther instance to any modern web browser and relays keyboard and mouse input back. Your flowgraph stays on the host with full access to its GPUs, compute backends, and attached radios, while your laptop, tablet, or phone only needs a browser.

Everything is completely self-hosted. There is no account, no cloud service, and no third-party relay.

```text
  Host Machine (CyberEther)                          Viewing Device (Browser)
  ┌─────────────────────────┐                        ┌──────────────────────┐
  │ Built-in Broker (:8686) │ ◀── Invite + Signal ─▶ │   /remote Viewer     │
  │ Native Engine + Encoder │ ◀═ Encrypted WebRTC ═▶ │ Video + Mouse/Keys   │
  └─────────────────────────┘                        └──────────────────────┘
```

## Quick Start

Remote Streaming supports both the Vulkan and Metal renderers, including headless mode. Linux and Windows default to Vulkan, while macOS defaults to Metal. On macOS, setting the encoder to Auto uses VideoToolbox hardware encoding whenever available. On Windows, Auto picks NVENC on NVIDIA, AMF on AMD, or QSV on Intel graphics. On Linux, Auto picks NVENC on NVIDIA or VAAPI on AMD and Intel graphics. With the Vulkan renderer, these hardware encoders read each frame straight from GPU memory whenever the driver supports sharing it, and fall back to copying frames through the CPU otherwise.

### Option 1: Start From the Desktop App

1. Open the **Start** tab (it appears automatically when no flowgraph is open, or press **Cmd/Ctrl + T** or click the **+** tab button).
2. In the **Remote** sidebar section, click **Start Streaming**.
3. Share access by clicking **Copy Link** or scanning the **QR Code** from a phone camera.
4. While waiting, the viewer shows a six-character code that also appears in the **Remote** viewer list. Confirm that both codes match, then click the check mark to approve the connection.

While streaming, click **Remote Sharing** in the titlebar to jump back to the Start tab. Click **Stop Streaming** to disconnect everyone, or click the **X** next to a specific viewer to remove only that device. Builds compiled without remote streaming hide the **Remote** section.

### Option 2: Start From a Headless Terminal

Launch CyberEther with `--remote` and `--headless`:

```bash
cyberether --remote --headless flowgraph.yml
```

The terminal prints a QR code, the invite link, and the access token. When a viewer requests access, the terminal prints its six-character code and waits for confirmation. Verify that it matches the viewer's screen, then type `y`.

In headless mode, the interface resizes to fit the browser window of the viewer in control, matching its pixel density up to 4K. The `--size` option only sets the size used before a viewer connects. Turn off **Match Viewer Resolution** to keep the launch size instead. A desktop window keeps its own size, and the stream follows it when the window is resized.

## How It Works

When streaming starts, CyberEther runs a lightweight broker on TCP port `8686` by default. That broker performs three tasks:

1. **Serves the viewer page** at `/remote`. The viewer page is fetched from the CyberEther CDN at build time and embedded into the binary, so nothing is downloaded from the internet at runtime.
2. **Hosts the signalling endpoint** at `/api/v1/remote/signaller` so the browser and host can exchange connection details.
3. **Manages the approval queue** so no viewer connects without permission.

Invite links take the form `http://<host>:8686/remote#<token>`. Because browsers never send the `#<token>` fragment in page requests, the token stays client-side until signalling begins. Anyone holding a valid invite can ask to join, though nobody enters until approved by the host (unless auto-approval is turned on).

Once approved, the broker steps out of the way. Video and input travel directly between host and browser over an end-to-end encrypted WebRTC connection.

## Network and Firewall Requirements

Because there is no cloud relay, the viewing device must have a direct network route to the host:

- **Same local network** works immediately with no extra setup.
- **Different networks** work seamlessly over a VPN such as WireGuard or Tailscale.
- **Hosts behind strict firewalls** that only allow SSH can carry everything through an SSH tunnel device (see [Streaming Over SSH](#streaming-over-ssh)).
- **Host firewalls** must permit the broker port (TCP `8686` by default) plus the UDP media ports. Configure **Media Port Range** to open a fixed range of ports instead of ephemeral ones.

> [!WARNING]
> The built-in broker is designed for trusted local networks. Never expose it directly to the public internet via port forwarding. Use a VPN or an SSH tunnel to connect from outside your network.

## Settings Reference

Adjust these options under **Settings** > **Remote**, or click the settings icon in the **Remote** sidebar panel on the Start tab. Saved changes apply the next time you start streaming. Every setting also has a matching command-line flag that overrides the saved value for one run only without touching `settings.yaml`.

### Access

| Setting | CLI Flag | Default | Description |
|---|---|---|---|
| **Viewer Approval** | `--remote-auto-approve <on\|off>` | Manual | Lets new viewers connect automatically or requires manual approval. |

Leave approval set to Manual unless every device that might see the invite link is trusted.

### Video

| Setting | CLI Flag | Default | Description |
|---|---|---|---|
| **Video Framerate** | `--remote-framerate <fps>` | `30` | Target frame rate of the outgoing stream. |
| **Video Codec** | `--remote-codec <codec>` | `H264` | Compression format (`h264`, `h265`, or `av1`). H265 and AV1 need an encoder that supports them, and H265 also needs a browser that can decode it. |
| **Video Encoder** | `--remote-encoder <encoder>` | `Auto` | Video encoder selection. Choosing `Auto` uses hardware encoding when available. |
| **Match Viewer Resolution** | `--remote-match-viewer <on\|off>` | On | Resizes a headless session to fit the browser window of the viewer in control. Turn off to keep the launch size. |

> [!TIP]
> H264 is the only codec every browser decodes. If a viewer connects without showing video after you pick `h265` or `av1`, switch **Video Codec** back to `h264`.

### Broker

| Setting | CLI Flag | Default | Description |
|---|---|---|---|
| **Broker Address** | `--remote-broker-address <ip>` | `0.0.0.0` | Interface address the broker binds to. The default listens on all network interfaces. |
| **Broker Port** | `--remote-broker-port <port>` | `8686` | Port serving `/remote` and the signalling channel. |
| **Broker Public URL** | `--remote-broker-public-url <url>` | Automatic | Origin placed into generated invite links. |
| **Broker TLS Certificate** | `--remote-broker-tls-cert <path>` | None | Path to a PEM certificate file (enables HTTPS). |
| **Broker TLS Key** | `--remote-broker-tls-key <path>` | None | Path to the matching PEM private key file. |

Bind **Broker Address** to a specific interface IP to limit access to one network, or set it to `127.0.0.1` to restrict access to the host itself.

When **Broker Public URL** is blank, CyberEther constructs invite links from the host's local IP address and broker port. Set it explicitly whenever viewers reach the host by another name, such as a TLS hostname or a VPN IP. Provide only the origin (`scheme://host:port`, like `https://cyberether.lan:8686`). Path prefixes are not supported because the viewer always connects at the root of the origin.

### Media

| Setting | CLI Flag | Default | Description |
|---|---|---|---|
| **Media Transport** | `--remote-media-transport <transport>` | `Auto` | Transport protocol offered for media (`auto` or `udp`). |
| **Media Port Range** | `--remote-media-port-range <min-max>` | Any | Fixed range of local ports for media, such as `40000-40100`. |
| **Media Address** | `--remote-media-address <ip>` | Automatic | Single local IP address advertised for media. Leave blank to advertise all addresses. |

Media always travels over UDP. On networks that block UDP, use a VPN or the SSH tunnel device described below.

Always allocate more than one port in **Media Port Range**. Each viewer consumes one port, and recently closed ports need a brief cooldown before reuse.

Setting **Media Address** restricts WebRTC candidates to one IP instead of advertising every host interface. This is primarily used with SSH tunnel devices, where you set it to the host's tunnel address.

## Securing the Connection With TLS

Out of the box, the broker serves plain HTTP. While the WebRTC video stream itself is always encrypted, the viewer page and signalling channel are not. On a shared or untrusted network, someone monitoring traffic could intercept the invite token or tamper with the viewer page. Enabling TLS protects the entire session.

Provide both **Broker TLS Certificate** and **Broker TLS Key** to switch the broker to HTTPS and secure WebSockets on the same port. Remember to set **Broker Public URL** to an `https://` origin matching the certificate's hostname, or browsers will reject the page.

Three straightforward ways to provision TLS:

- **Local certificate authority.** Tools like `mkcert` issue trusted certificates for local hostnames or IP addresses across your own devices without browser warnings.
- **Public domain certificate.** If the host has a DNS name, a certificate from a public authority works automatically on every device.
- **Reverse proxy.** Place Caddy or nginx in front of the broker to terminate TLS. Forward the entire origin including WebSocket upgrades, and point **Broker Public URL** at the proxy. Path prefixes are not supported.

Always restrict file permissions so only the user running CyberEther can read the private key.

## Streaming Over SSH

If a remote host sits behind a firewall that only permits SSH, an SSH tunnel device can carry both the broker and the media stream. Media uses UDP, which plain SSH port forwarding cannot carry, so the tunnel must be a layer 3 device created with `ssh -w`.

### Step 1: Allow Tunnel Devices on the Host

Enable tunnels in the host's `/etc/ssh/sshd_config` and restart the SSH server:

```
PermitTunnel yes
```

Creating the tunnel device needs root on both machines, or an account allowed to create TUN devices.

### Step 2: Open the Tunnel

From your viewing machine, open the tunnel and give each end an address:

```bash
sudo ssh -N -w 0:0 root@host
```

Then, in another terminal on each machine, bring the device up:

```bash
# On the host
sudo ip addr add 10.10.0.1/30 dev tun0
sudo ip link set tun0 up

# On the viewing machine
sudo ip addr add 10.10.0.2/30 dev tun0
sudo ip link set tun0 up
```

### Step 3: Launch CyberEther on the Host

Bind the broker and the media to the tunnel address:

```bash
cyberether --remote --headless \
  --remote-broker-address 10.10.0.1 \
  --remote-broker-public-url http://10.10.0.1:8686 \
  --remote-media-address 10.10.0.1 \
  flowgraph.yml
```

You can also save these values permanently under **Settings** > **Remote**. Run this inside an interactive SSH session so approval prompts appear in your terminal, or add `--remote-auto-approve on` since only users with tunnel access can reach the broker.

Once the tunnel is up, open the invite link from the host (such as `http://10.10.0.1:8686/remote#<token>`) in a browser on your viewing machine.

### Practical Notes for SSH Tunneling

- The commands above use Linux `ip`. Other systems name and configure tunnel devices differently, and some SSH clients, including the one bundled with macOS, do not support `-w`. A VPN such as WireGuard or Tailscale is the simpler choice on those systems.
- Expect slightly higher latency than on a direct LAN connection because SSH carries every packet over a single TCP connection and adds encryption overhead.
- Reducing **Video Framerate** keeps the session responsive over slower links.
- Because the broker only listens on the tunnel address, nothing is exposed on the host's network, and SSH encrypts all traffic including the viewer page and invite token.

## Troubleshooting

| Symptom | What to Check |
|---|---|
| **Invite link will not open** | Confirm both devices are on the same network (or VPN), check the host firewall for the broker port, and set **Broker Public URL** if the host has multiple network interfaces. |
| **Viewer stays on the waiting screen** | The host has not approved the connection yet. Match the six-character code in the **Remote** sidebar panel or terminal. |
| **Viewer connects with a black screen** | Switch **Video Codec** to `h264`. On strict networks, verify that your UDP media ports are open or use a VPN. |
| **Streaming fails to start** | Another process may already own port `8686` (choose another **Broker Port**), your TLS certificate and key files may be missing or mismatched, or your default renderer may be unavailable (try `--renderer vulkan` or `--renderer metal`). |
